"""Per-arc relative variance plus shared monthly temporal covariance.
Precision = prior_precision / variance_factor * inverse(monthly_covariance).
Arc VCE follows GROOPS PreprocessingSst: whitened RSS / redundancy, including
both shared-local and global leverage. Global trace uses reproducible probes.
"""
import copy
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from scipy.fft import rfft,irfft,next_fast_len,dct,idct
from tqdm import tqdm
from .core import ObservationBlock
from .parallel import map_arcs

DEFAULTS=dict(Mode='Off',PriorPrecision={},VarianceFactors={},GroupVarianceFactors={},RedundancyProbes=64,
              RandomSeed=20240724,MinRedundancy=1.,MinVariance=.25,MaxVariance=16.,
              UpdateDamping=.5,MaxAdjustment=4.,WeightTolerance=.02,CovarianceTolerance=.02)


def weight_options(solver):
    c=dict(DEFAULTS);c.update(solver.get('ArcWeightingConfig',{}))
    if c['Mode'] not in ('Off','Fixed','VCE'):raise ValueError('ArcWeighting Mode must be Off, Fixed or VCE')
    if (not isinstance(c['RedundancyProbes'],int) or c['RedundancyProbes']<1
        or not 0<c['MinVariance']<=1<=c['MaxVariance'] or not 0<c['UpdateDamping']<=1
        or c['MaxAdjustment']<1 or c['MinRedundancy']<=0 or c['WeightTolerance']<=0 or c['CovarianceTolerance']<=0):
        raise ValueError('invalid arc variance estimation controls')
    for table in ('PriorPrecision','VarianceFactors','GroupVarianceFactors'):
        if not isinstance(c[table],dict):raise ValueError(table+' must map observation names to arc IDs')
        if set(c[table])-{'Orbit_A','Orbit_B','RangeRate'}:raise ValueError('unknown observation type in '+table)
        for values in c[table].values():
            if table=='GroupVarianceFactors' and isinstance(values,dict):
                raise ValueError('GroupVarianceFactors requires one scalar per observation type')
            numbers=values.values() if isinstance(values,dict) else [values]
            if any(not np.isfinite(v) or v<=0 for v in numbers):raise ValueError(table+' must contain positive finite values')
    return c


def lookup(table,name,arc):
    v=table.get(name,1.)
    return float(v.get(str(arc),1.) if isinstance(v,dict) else v)


def precision_factor(solver,arc,name):
    c=weight_options(solver)
    if c['Mode']=='Off':return 1.
    return lookup(c['PriorPrecision'],name,arc)/(lookup(c['VarianceFactors'],name,arc)*lookup(c['GroupVarianceFactors'],name,arc))


def covariance_probes(normal,count,seed):
    scale=np.sqrt(np.diag(normal));n=normal/scale[:,None]/scale[None,:]
    l=cholesky(n,lower=True)
    z=np.random.default_rng(seed).choice([-1.,1.],size=(len(scale),count))/np.sqrt(count)
    return solve_triangular(l.T,z,lower=False)/scale[:,None]


def block_statistics(blocks,local_global,global_delta,local_delta,probes):
    white=[block.whiten() for block in blocks]
    b=np.concatenate([x[1] for x in white])
    if b.shape[1]:
        q,_=np.linalg.qr(b/np.linalg.norm(b,axis=0),mode='reduced')
    else:q=np.empty((b.shape[0],0))
    result={};start=0;kp=local_global@probes
    for block,(a,b,y) in zip(blocks,white):
        count=len(y);local_leverage=float(np.sum(q[start:start+count]**2))
        az=a@probes-b@kp
        global_leverage=float(np.sum(az**2))
        e=y-a@global_delta-b@local_delta
        result[block.name]=dict(weighted_rss=float(e@e),observations=count,
            local_leverage=local_leverage,global_leverage=global_leverage,
            redundancy=count-local_leverage-global_leverage,precision_weight=block.weight)
        start+=count
    return result


def update_variances(solver,statistics):
    c=copy.deepcopy(weight_options(solver));previous=copy.deepcopy(c['VarianceFactors'])
    records=[];group_changes=[]
    for arc,groups in statistics.items():
        for name,st in groups.items():
            old=lookup(previous,name,arc);red=st['redundancy']
            eligible=red>=c['MinRedundancy'] and np.isfinite(red)
            value=old
            if eligible:
                ratio=max(st['weighted_rss']/red,np.finfo(float).tiny)
                step=np.clip(ratio,1/c['MaxAdjustment'],c['MaxAdjustment'])**c['UpdateDamping']
                value=old*step
            records.append(dict(arc=str(arc),observation=name,old=old,new=float(value),
                                eligible=bool(eligible),**st))
    for name in ('Orbit_A','Orbit_B','RangeRate'):
        group=[r for r in records if r['observation']==name and r['eligible']]
        if group:
            # Median sets the typical arc to one; absolute amplitude belongs to
            # the monthly covariance, avoiding a non-identifiable double scale.
            center=float(np.median([r['new'] for r in group]))
            if name!='RangeRate' and solver.get('OrbitCovarianceMode') in ('ITSGEpoch','FixedDiagonal'):
                c['GroupVarianceFactors'][name]=lookup(c['GroupVarianceFactors'],name,0)*center
                group_changes.append(abs(np.log(center)))
            values={str(r['arc']):lookup(previous,name,r['arc']) for r in records if r['observation']==name}
            for r in group:
                r['new']=float(np.clip(r['new']/center,c['MinVariance'],c['MaxVariance']))
                values[r['arc']]=r['new']
            c['VarianceFactors'][name]=values
    change=float(max([abs(np.log(r['new']/r['old'])) for r in records if r['eligible']]+group_changes+[0.]))
    return c,dict(max_log_variance_change=change,converged=change<=c['WeightTolerance'],records=records,
                  trace_method='Monte Carlo global leverage; exact local QR leverage',
                  variance_normalization='median per observation group')


def read_statistics(solution):
    root=Path(solution);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    result={}
    for arc in manifest['arcs']:
        with h5py.File(root/('arc_%d.h5'%arc),'r') as h:
            result[str(arc)]={name:{k:(v.item() if isinstance(v,np.generic) else v) for k,v in h[name].attrs.items()} for name in ('Orbit_A','Orbit_B','RangeRate') if name in h}
    return result


def pooled_autocovariance(arcs,dt,max_lag,zero_frequency_policy="Keep"):
    """Pool uncentered residual products; retain arc means and trends."""
    if zero_frequency_policy not in ("Keep", "CopyFirstNonzero"):
        raise ValueError("invalid CovarianceZeroFrequencyPolicy")
    numerator=np.zeros(max_lag+1);pairs=np.zeros(max_lag+1)
    for time,residual in arcs:
        t=np.asarray(time,float);x=np.asarray(residual,float).reshape(-1)
        if t.shape!=x.shape or not len(t) or not np.isfinite(x).all():raise ValueError('invalid covariance arc')
        coordinate=(t-t[0])/dt;index=np.rint(coordinate).astype(int)
        if not np.allclose(coordinate,index,rtol=0,atol=1e-6) or len(np.unique(index))!=len(index):
            raise ValueError('covariance observations must lie on the specified physical grid')
        grid=np.zeros(index[-1]+1);mask=np.zeros(len(grid));grid[index]=x;mask[index]=1
        length=next_fast_len(2*len(grid));fx=rfft(grid,length);fm=rfft(mask,length)
        stop=min(max_lag+1,len(grid))
        numerator[:stop]+=irfft(fx*fx.conj(),length)[:stop]
        pairs[:stop]+=np.maximum(0,np.rint(irfft(fm*fm.conj(),length)[:stop]))
    covariance=np.divide(numerator,pairs,out=np.zeros_like(numerator),where=pairs>0)
    if covariance[0]<=0 or not np.isfinite(covariance).all():raise ValueError('degenerate monthly residual covariance')
    covariance*=1-np.arange(max_lag+1)/(max_lag+1.)
    spectrum=dct(covariance,type=1)
    spectrum=np.maximum(spectrum,covariance[0]*1e-4)
    if zero_frequency_policy == "CopyFirstNonzero":
        if spectrum.size < 2:raise ValueError("zero-frequency copy requires a nonzero frequency bin")
        spectrum[0]=spectrum[1]
    covariance=idct(spectrum,type=1)
    covariance[0]+=max(covariance[0]*1e-10,1e-30)
    return covariance,pairs


def _noise_arc_job(arc,source,solver):
    from .channel import Settings,epoch_indices
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload
    from src.Solver.OrbitObservations import epoch_covariance
    source=Path(source);settings=Settings(source/'settings',source)
    data={}
    mode=solver.get('OrbitCovarianceMode','EstimatedFull')
    from .initial_weights import wls_basis, basis_residual
    basis = wls_basis(solver)
    rd=ArcData(settings.obj('InterfaceConfig')) if mode=='ITSGEpoch' or basis else None
    with h5py.File(source/('arc_%d.h5'%arc),'r') as h:
        for name in ('Orbit_A','Orbit_B','RangeRate'):
            if name not in h:continue
            t=h[name+'/time'][()];res=h[name+'/linear_postfit'][()]
            if name!='RangeRate':res=res.reshape(-1,3)
            if mode=='ITSGEpoch' and name!='RangeRate':
                obs=rd.getData(arc,Payload.KinematicOrbit,name[-1]);idx=epoch_indices(obs[:,0],t)
                res=np.linalg.solve(np.linalg.cholesky(epoch_covariance(obs[idx,4:10])),res[:,:,None])[:,:,0]
            if basis:
                components = None
                if name != 'RangeRate':
                    obs=rd.getData(arc,Payload.KinematicOrbit,name[-1]);idx=epoch_indices(obs[:,0],t)
                    components=obs[idx,4:10]
                res=basis_residual(res,name,solver,components)
            res=res*np.sqrt(precision_factor(solver,arc,name))
            data[name]=(t,res)
    return data


def _model_change(filename,autocovariance=None,diagonal=None):
    if not Path(filename).is_file():return None
    with h5py.File(filename,'r') as h:
        if diagonal is not None:
            old=h.attrs.get('diagonal_value')
            return None if old is None else float(abs(diagonal-old)/max(abs(old),1e-30))
        if 'autocovariance' not in h:return None
        old=h['autocovariance'][()]
        if old.shape!=autocovariance.shape:return None
        return float(np.linalg.norm(autocovariance-old)/max(np.linalg.norm(old),1e-30))


def collect_noise_groups(solution,solver,workers):
    root=Path(solution);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    groups={name:[] for name in ('Orbit_A','Orbit_B','RangeRate')}
    for data in tqdm(map_arcs(_noise_arc_job,manifest['arcs'],(root,solver),workers),
                      total=len(manifest['arcs']),desc='Pool covariance within arcs',unit='arc'):
        for name,value in data.items():groups[name].append(value)
    return groups


def estimate_monthly_noise(settings,solution,gls,workers):
    from .channel import write_json
    from src.Solver.CovarianceModel import save_stationary_covariance,orbit_covariance_filename,range_rate_covariance_filename
    root=Path(solution);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    solver=settings.raw['SolverConfig']
    groups=collect_noise_groups(solution,solver,workers)
    if solver.get('SaveIterationProducts',False):
        from .iteration_products import save_input_psds
        save_input_psds(groups,solution,solver)
    mode=gls.OrbitCovarianceMode;metadata={};changes={}
    for name,values in groups.items():
        if not values:continue
        filename=(range_rate_covariance_filename(solver['PostKBRR'],settings.span) if name=='RangeRate'
                  else orbit_covariance_filename(solver['PostOrbit'],settings.span,name[-1]))
        if mode=='FixedDiagonal' and name!='RangeRate':
            changes[name]=_model_change(filename,diagonal=float(gls.OrbitFixedDiagonal))
            save_stationary_covariance(filename,diagonal_value=float(gls.OrbitFixedDiagonal));continue
        intervals=np.concatenate([np.diff(t) for t,_ in values if len(t)>1])
        if len(intervals)==0:raise ValueError('need at least two epochs to estimate covariance')
        dt=float(np.min(intervals[intervals>0]))
        configured=solver.get('CovarianceMaxLagSeconds')
        duration=float(settings.raw['InterfaceConfig']['arc_length'])*3600
        max_lag=max(1,int((duration if configured is None else float(configured))/dt))
        components=1 if name=='RangeRate' else 3
        covariances=[]
        for axis in range(components):
            acov,pairs=pooled_autocovariance([(t,r if components==1 else r[:,axis]) for t,r in values],dt,max_lag,solver.get("CovarianceZeroFrequencyPolicy","Keep"))
            covariances.append(acov)
        changes[name]=_model_change(filename,autocovariance=np.asarray(covariances))
        save_stationary_covariance(filename,autocovariance=np.asarray(covariances),sample_interval=dt,
                                   time_origin=min(t[0] for t,_ in values))
        metadata[name]=dict(dt=dt,lag_count=max_lag+1,pair_count=pairs.tolist(),arc_count=len(values))
    tolerance=weight_options(solver)['CovarianceTolerance']
    return dict(zero_frequency_policy=solver.get("CovarianceZeroFrequencyPolicy","Keep"), remove_arc_mean=False, remove_trend=False, residual_space=solver.get('GLSResidualSpace','Physical'), estimator='pooled FFT autocovariance, no cross-arc pairs or gap filling',models=metadata,
                relative_covariance_changes=changes,
                converged=all(v is not None and v<=tolerance for v in changes.values()))
