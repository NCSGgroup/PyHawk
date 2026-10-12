"""Position observations are independent of the GNV propagation seed grid."""
import numpy as np
from src.Preference.EnumType import Payload


def orbit_payload(config, fallback=Payload.GNV):
    name=config.get('ObservationSource')
    if name is None:
        return fallback
    aliases={'GNV':Payload.GNV,'KIN':Payload.KinematicOrbit,'KINEMATICORBIT':Payload.KinematicOrbit}
    try:return aliases[str(name).upper()]
    except KeyError:raise ValueError('ObservationSource must be GNV or KIN')


def align_positions(observations, nominal_time, sampling=0., sampling_origin=0.):
    obs=np.asarray(observations,float);time=np.asarray(nominal_time,float)
    if obs.ndim!=2 or obs.shape[1]<4 or obs.shape[0]==0:
        raise ValueError('missing position observations; no automatic GNV fallback')
    if not np.isfinite(time).all() or time.ndim!=1 or len(time)==0:
        raise ValueError('invalid propagation epochs')
    keys=np.rint(time*1e6).astype(np.int64)
    if np.any(np.diff(keys)<=0):raise ValueError('propagation epochs must be strictly increasing')
    valid=np.isfinite(obs[:,:4]).all(axis=1)
    selected=np.flatnonzero(valid)
    if sampling<0 or not np.isfinite(sampling):raise ValueError('invalid orbit sampling')
    if sampling:
        if len(selected)==0:raise ValueError('no finite orbit positions')
        origin=obs[selected,0].min() if sampling_origin=='FirstEpoch' else (0. if sampling_origin=='GPS' else float(sampling_origin))
        if not np.isfinite(origin):raise ValueError('invalid orbit sampling origin')
        coordinate=(obs[selected,0]-origin)/sampling
        selected=selected[np.isclose(coordinate,np.rint(coordinate),rtol=0,atol=1e-8)]
    targets=np.rint(obs[selected,0]*1e6).astype(np.int64)
    if len(np.unique(targets))!=len(targets):raise ValueError('duplicate observation epochs')
    idx=np.searchsorted(keys,targets);inside=idx<len(keys)
    matched=np.zeros(len(idx),bool);matched[inside]=keys[idx[inside]]==targets[inside]
    selected,idx=selected[matched],idx[matched]
    if len(idx)==0:raise ValueError('no position observations coincide with propagation epochs')
    order=np.argsort(idx)
    return selected[order],idx[order]


def epoch_covariance(components):
    x=np.asarray(components,float)
    if x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all():
        raise ValueError('KIN requires finite [xx,yy,zz,xy,xz,yz] covariance columns')
    c=np.zeros((len(x),3,3))
    c[:,0,0],c[:,1,1],c[:,2,2]=x[:,0],x[:,1],x[:,2]
    c[:,0,1]=c[:,1,0]=x[:,3];c[:,0,2]=c[:,2,0]=x[:,4];c[:,1,2]=c[:,2,1]=x[:,5]
    try:np.linalg.cholesky(c)
    except np.linalg.LinAlgError as error:raise ValueError('KIN epoch covariance is not positive definite') from error
    return c


def fit_position_increment(observations,time,state,epoch_weighting=False,sampling=0.,sampling_origin=0.):
    rows,idx=align_positions(observations,time,sampling,sampling_origin)
    obs=np.asarray(observations)[rows];xyz=np.asarray(state)[idx]
    residual=obs[:,1:4]-xyz[:,:,0]
    design=xyz[:,:,1:]
    if epoch_weighting:
        if obs.shape[1]<10:raise ValueError('KIN fit requested epoch covariance, but six columns are absent')
        l=np.linalg.cholesky(epoch_covariance(obs[:,4:10]))
        design=np.linalg.solve(l,design);residual=np.linalg.solve(l,residual[:,:,None])[:,:,0]
    matrix=design.reshape(-1,design.shape[-1])
    if matrix.shape[0]<=matrix.shape[1]:raise ValueError('too few KIN positions for orbit/ACC fit')
    scale=np.linalg.norm(matrix,axis=0)
    if not np.isfinite(matrix).all() or np.any(scale==0):
        raise ValueError('position fit has invalid or unobservable parameter columns')
    delta,_,rank,_=np.linalg.lstsq(matrix/scale,residual.reshape(-1),rcond=None)
    if rank!=matrix.shape[1]:raise ValueError('position fit design is rank deficient')
    return delta/scale
