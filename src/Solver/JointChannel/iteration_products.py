"""Per-solution GFC and uncentered, gap-aware input residual PSD diagnostics."""
from pathlib import Path
import json
import numpy as np
from scipy.fft import rfft, rfftfreq, next_fast_len


def pooled_periodogram(arcs):
    """Boxcar spectra, pooled by observed sample count; no cross-arc products.

    Missing grid entries contribute no samples. The observation mask therefore
    defines the spectral window; this is not an interpolation or gap correction.
    """
    intervals=np.concatenate([np.diff(t) for t,x in arcs if len(t)>1])
    dt=float(np.min(intervals[intervals>0]))
    rows=[]
    for t,x in arcs:
        t=np.asarray(t,float);x=np.asarray(x,float)
        if x.ndim==1:x=x[:,None]
        index=np.rint((t-t[0])/dt).astype(int)
        if not np.allclose((t-t[0])/dt,index,rtol=0,atol=1e-6) or len(np.unique(index))!=len(index):
            raise ValueError('PSD requires unique epochs on the covariance sampling grid')
        if len(x)!=len(t) or not np.isfinite(x).all():raise ValueError('invalid PSD residuals')
        rows.append((index,x))
    nfft=next_fast_len(max(index[-1]+1 for index,x in rows))
    total=np.zeros((nfft//2+1,rows[0][1].shape[1]));count=0
    for index,x in rows:
        grid=np.zeros((nfft,x.shape[1]));grid[index]=x
        total+=abs(rfft(grid,axis=0))**2;count+=len(x)
    total*=dt/count
    total[1:-1 if nfft%2==0 else None]*=2
    return rfftfreq(nfft,dt),total,dict(dt=dt,nfft=nfft,samples=count,arcs=len(arcs),
        demean=False,detrend=False,window='boxcar',pooling='observed-sample-count',
        gaps='masked Fourier sum; no interpolation; spectral window includes gaps')


def save_input_psds(groups,solution,solver,purpose='covariance_input'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(solution)/'diagnostics'/purpose;root.mkdir(parents=True,exist_ok=True)
    metadata=dict(purpose=purpose,source_solution=str(solution),
        residual='linear_postfit after the same basis and precision-factor scaling as covariance estimation',
        residual_space=solver.get('GLSResidualSpace','Physical'),
        arc_weighting=solver.get('ArcWeightingConfig',{}),
        spectrum_before_lag_taper_psd_floor_and_zero_frequency_copy=True,groups={})
    for name,values in groups.items():
        if not values:continue
        f,p,info=pooled_periodogram(values)
        np.savez_compressed(root/(name+'_PSD.npz'),frequency_hz=f,psd=p)
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        labels=['KBRR'] if p.shape[1]==1 else ['X','Y','Z']
        for i,label in enumerate(labels):
            ax.loglog(f[1:],np.maximum(p[1:,i],np.finfo(float).tiny),label=label,lw=1)
        ax.set_xlabel('Frequency [Hz]')
        units='(m/s)^2/Hz' if name=='RangeRate' else 'm^2/Hz'
        if solver.get('GLSResidualSpace','Physical')!='Physical':units='transformed units^2/Hz'
        ax.set_ylabel('PSD ['+units+']')
        ax.set_title(Path(solution).name+' | '+name)
        ax.legend(frameon=False);ax.grid(True,which='major',alpha=.2)
        fig.savefig(root/(name+'_PSD.png'),dpi=180);fig.savefig(root/(name+'_PSD.pdf'));plt.close(fig)
        info['zero_frequency_psd']=p[0].tolist();info['zero_frequency_not_shown_on_log_axis']=True
        metadata['groups'][name]=info
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Residual PSD saved before covariance estimation: '+str(root),flush=True)
    return root


def export_iteration_gfc(settings,solution,cumulative,workers=1):
    from .workflow import formal_covariance,write_json
    from src.ForceModel.RefGravModel import RefGravModel
    from src.Auxilary.SortCS import SortCS
    from src.Auxilary.StandardSH import StandardSH
    sh=settings.raw['GravityDesignParameterConfig']['StokesCoefficientsConfig']
    skip=18 if settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']['isScale'] else 0
    force=settings.obj('ForceModelConfig');force.resolve_time_variable_date(settings.span)
    reference=RefGravModel().configure(force).getStaticModel()
    covariance,statistics=formal_covariance(settings,solution,workers)
    sort=SortCS(sh['MaxDegree'],sh['MinDegree'],sh['SortMethod'])
    c,s=sort.invert(np.asarray(cumulative)[skip:]);rc,rs=reference.getCS(sh['MaxDegree'])
    sc,ss=sort.invert(np.sqrt(np.maximum(np.diag(covariance)[skip:],0)))
    out=StandardSH().configure(str(Path(solution)/'products'),settings.span,sh['MaxDegree'],
        c+rc,s+rs,sc,ss,mission=settings.raw['InterfaceConfig']['mission'],reference_model=reference,
        unused_days=settings.raw['InterfaceConfig'].get('date_delete',[])).unfiltered()
    write_json(Path(solution)/'iteration_products.json',dict(SH=str(out),statistics=statistics,
        convention='reference plus cumulative gravity; monthly ACC scale excluded'))
    return out
