"""Conservative validation for reuse of completed monthly-scale arc integrations."""
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np
from .channel import Settings


def input_fingerprints(settings, arcs):
    """Track shared mutable preprocessing/fitting inputs without hashing GB matrices."""
    raw = Path(settings.raw['InterfaceConfig']['PathOfFilesConfig']['temp_raw_data'])/(settings.tag+'.hdf5')
    fit = Path(settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['OrbitAdjustResTemp'])/settings.tag
    paths = [raw] + [fit/('%d_AB.hdf5'%a) for a in arcs]
    result = {}
    for p in paths:
        st = p.stat()
        value = dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
        if p != raw:
            value['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        result[str(p)] = value
    return result


def validate_bootstrap_settings(settings, cache):
    saved = Settings(cache/'settings', cache)
    # Solver weighting and output destinations do not change the dynamics.
    for key in saved.raw:
        if key not in ('SolverConfig','NEQConfig') and saved.raw[key] != settings.raw[key]:
            raise ValueError('bootstrap cache dynamics settings differ: '+key)
    manifest = cache/'manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if input_fingerprints(settings,data['arcs']) != data['inputs']:
            raise ValueError('bootstrap cache inputs changed since checkpoint')
    else:
        # Legacy checkpoints lack provenance hashes: reject inputs newer than
        # the saved settings, then verify per-arc states and ACC force numerically.
        stamp = max(p.stat().st_mtime_ns for p in (cache/'settings').glob('*.json'))
        raw = Path(settings.raw['InterfaceConfig']['PathOfFilesConfig']['temp_raw_data'])/(settings.tag+'.hdf5')
        if raw.stat().st_mtime_ns > stamp:
            raise ValueError('legacy bootstrap raw observations changed after snapshot')


def reuse_bootstrap_arc(settings, arc, cache, root, initial, acc):
    from src.Interface.Accelerometer import AccCaliPar_V3, Accelerometer_V3
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload
    from .feedback_cache import reuse_integration
    folder = cache/'DesignMatrix'/settings.tag
    if not all((folder/n).exists() for n in (
            '%d_AB.hdf5'%arc,'Orbit_%d_AB.hdf5'%arc,'RangeRate_%d.hdf5'%arc)):
        return False
    checkpoint = cache/'checkpoints'/('arc_%d.json'%arc)
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if not np.array_equal(saved['initial'],initial) or any(
                not np.array_equal(saved['acc'][i],acc[i]) for i in range(2)):
            raise ValueError('bootstrap cache physical parameters changed for arc %d'%arc)
    else:
        stamp = max(p.stat().st_mtime_ns for p in (cache/'settings').glob('*.json'))
        fit = Path(settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['OrbitAdjustResTemp'])/settings.tag/('%d_AB.hdf5'%arc)
        if fit.stat().st_mtime_ns > stamp:
            raise ValueError('legacy fit changed after bootstrap snapshot for arc %d'%arc)
        with h5py.File(cache/'state.h5','r') as h:
            key = 'arc_%d'%arc
            if key in h:
                if not np.array_equal(h[key+'/initial'][()],initial) or any(
                        not np.array_equal(h[key+'/acc_'+sat][()],acc[i]) for i,sat in enumerate('AB')):
                    raise ValueError('legacy saved state differs for arc %d'%arc)
    rd = ArcData(settings.obj('InterfaceConfig'))
    p = settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']
    force_files = []
    try:
        for i,sat in enumerate('AB'):
            ap = AccCaliPar_V3(rd,sat,arc,p['X_AdjustLength'],p['Y_AdjustLength'],p['Z_AdjustLength'],
                              settings.obj('DesignAccelerometerConfig_'+sat))
            model = Accelerometer_V3(ap).updatePar(acc[i]).get_acceleration()
            f = cache/'NonConservativeForce'/(settings.tag+'_%s_%d.hdf5'%(sat,arc))
            with h5py.File(f,'r') as h:
                if h['acc'].shape != model.shape or not np.allclose(h['acc'][()],model,rtol=1e-12,atol=1e-18):
                    raise ValueError('cached ACC force does not match arc %d'%arc)
            force_files.append(f)
        # Match the exact initial epoch used by Orbit_var_2nd_diff.setInitData.
        a,b = (rd.getData(arc,Payload.GNV,sat) for sat in 'AB')
        epochs = np.intersect1d(a[:,0],b[:,0]) if len(a)==len(b) else a[:,0]
        initial_time = epochs[4]
        with h5py.File(folder/('%d_AB.hdf5'%arc),'r') as raw, h5py.File(folder/('Orbit_%d_AB.hdf5'%arc),'r') as orbit, h5py.File(folder/('RangeRate_%d.hdf5'%arc),'r') as rate:
            t = raw['t'][()]
            hit = np.flatnonzero(np.abs(t-initial_time)<1e-6)
            if len(hit)!=1:
                raise ValueError('missing initial epoch in cached arc %d'%arc)
            k=int(hit[0])
            if not np.allclose(raw['r'][k,:,0],initial[:,:3].reshape(-1),rtol=0,atol=1e-7) or not np.allclose(raw['v'][k,:,0],initial[:,3:].reshape(-1),rtol=0,atol=1e-10):
                raise ValueError('cached trajectory initial state mismatch for arc %d'%arc)
            nl = 6+len(acc[0])-9
            ng = settings.raw['GravityDesignParameterConfig']['StokesCoefficientsConfig']['Parameter_Number']+9
            for key in ('r','v'):
                if raw[key].shape!=(len(t),6,1+nl+ng) or orbit[key].shape!=(len(t),6):
                    raise ValueError('cached state shape mismatch')
                if orbit['local_'+key].shape!=(len(t),6,nl) or orbit['global_'+key].shape!=(len(t),6,ng):
                    raise ValueError('cached design shape mismatch')
            if rate['global'].shape!=(len(t),ng+9) or rate['local'].shape!=(len(t),2*nl):
                raise ValueError('cached KBRR shape mismatch')
    except (OSError,KeyError,ValueError) as exc:
        print('Bootstrap cache rejected arc %d: %s; reintegrating'%(arc,exc),flush=True)
        return False
    # Full finite-value scan of raw and derived matrices before creating links.
    reused = reuse_integration(settings,arc,cache,root)
    if reused is None:
        print('Bootstrap cache rejected arc %d: matrix validation failed'%arc,flush=True)
        return False
    dest=root/'NonConservativeForce';dest.mkdir(exist_ok=True)
    for f in force_files:
        os.link(str(f),str(dest/f.name))
    return True