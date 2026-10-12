"""One absolute 3x3 ACC scale matrix per satellite and processing span."""
import traceback
from time import perf_counter
from contextlib import redirect_stderr
from pathlib import Path
import h5py
import numpy as np
from .channel import Settings, write_json
from .parallel import map_arcs


def configure_monthly_scale(settings):
    parameter = settings.raw['GravityDesignParameterConfig']
    p = parameter['AccelerometerConfig']
    transition = parameter['TransitionMatrixConfig']
    if (not p['isRequired'] or p['AllisGlobal'] or not transition['isRequired']
            or transition['Parameter_Number'] != 6 or transition['AllisGlobal']):
        raise ValueError('monthly scale requires local initial6/bias and estimated ACC')
    fit = settings.raw['OrbitFitParameterConfig']['AccelerometerConfig']
    for key in ('X_AdjustLength', 'Y_AdjustLength', 'Z_AdjustLength'):
        if p[key] != fit[key]:
            raise ValueError('monthly scale bootstrap requires identical fit/design bias segmentation')
    for sat in ('A', 'B'):
        source = settings.raw['AccelerometerConfig_'+sat]
        design = settings.raw['DesignAccelerometerConfig_'+sat]
        for key in ('Bias_constant', 'Bias_trend', 'Bias_quadratic'):
            if (source['cali'][key] != design['cali'][key]
                    or source['cali_value'][key] != design['cali_value'][key]):
                raise ValueError('monthly scale bootstrap requires matching bias layout and fixed values')
        scale = np.asarray(design['cali_value']['Scale'], float)
        if scale.shape != (9,) or not np.isfinite(scale).all():
            raise ValueError('monthly scale requires nine finite configured matrix elements')
        design['cali']['Scale'] = [True]*9
    p['isScale'] = True


def bootstrap_parameters(fitted, source_scale_count, target_width, common_scale):
    fitted = np.asarray(fitted, float)
    common_scale = np.asarray(common_scale, float)
    if (fitted.ndim != 1 or common_scale.shape != (9,)
            or fitted.size-source_scale_count != target_width-9):
        raise ValueError('saved fit and monthly-scale bias widths differ')
    result = np.concatenate((common_scale, fitted[source_scale_count:]))
    if not np.isfinite(result).all():
        raise ValueError('non-finite bootstrap ACC state')
    return result


def validate_shared_scale(acc, common):
    common = np.asarray(common, float)
    if common.shape != (2,9) or not np.isfinite(common).all():
        raise ValueError('invalid shared monthly scale state')
    for i in range(2):
        if not np.allclose(np.asarray(acc[i])[:9], common[i], rtol=0, atol=1e-14):
            raise ValueError('arc absolute scale differs from shared monthly scale')


def _bootstrap_arc_job(arc, root, cache=None):
    from src.Interface.Accelerometer import AccCaliPar_V3
    from src.Interface.ArcOperation import ArcData
    from .feedback import integrate_state
    started = perf_counter()
    root = Path(root)
    settings = Settings(root/'settings', root)
    rd = ArcData(settings.obj('InterfaceConfig'))
    fit = settings.raw['OrbitFitParameterConfig']['AccelerometerConfig']
    p = settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']
    filename = (Path(settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['OrbitAdjustResTemp'])
                / settings.tag / ('%d_AB.hdf5'%arc))
    with h5py.File(filename, 'r') as h:
        initial, fitted = h['TransitionMatrix'][()], h['Accelerometer'][()]
    acc = []
    for i,sat in enumerate(('A','B')):
        source = settings.obj('AccelerometerConfig_'+sat)
        ap = AccCaliPar_V3(rd,sat,arc,fit['X_AdjustLength'],fit['Y_AdjustLength'],fit['Z_AdjustLength'],source)
        if fitted[i].size != ap.getPar().size:
            raise ValueError('saved fit width differs from fitting config; restore its original masks')
        design = settings.obj('DesignAccelerometerConfig_'+sat)
        target = AccCaliPar_V3(rd,sat,arc,p['X_AdjustLength'],p['Y_AdjustLength'],p['Z_AdjustLength'],design)
        acc.append(bootstrap_parameters(fitted[i],sum(source.cali['Scale']),
                   target.getPar().size,design.cali_value['Scale']))
    sh = settings.raw['GravityDesignParameterConfig']['StokesCoefficientsConfig']
    # SortCS ordering contains (Nmax+1)^2-Nmin^2 real coefficients.
    gravity = np.zeros((sh['MaxDegree']+1)**2-sh['MinDegree']**2)
    reused = False
    if cache is not None:
        from .bootstrap_cache import reuse_bootstrap_arc
        reused = reuse_bootstrap_arc(settings, arc, Path(cache), root, initial, acc)
    if not reused:
        logdir = root/'logs'
        logdir.mkdir(exist_ok=True)
        print('Bootstrap integrate arc %d' % arc, flush=True)
        try:
            # Persist worker tqdm and exceptions instead of discarding stderr.
            with open(logdir/('arc_%d.err'%arc), 'w') as log, redirect_stderr(log):
                try:
                    integrate_state(settings,arc,initial,acc,gravity,root)
                except Exception:
                    traceback.print_exc()
                    raise
        except Exception as exc:
            print('Bootstrap FAILED arc %d: %r (see %s)' % (arc,exc,logdir),flush=True)
            raise
    checkpoint = root/'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    write_json(checkpoint/('arc_%d.json'%arc), dict(arc=int(arc),initial=initial.tolist(),
               acc=[a.tolist() for a in acc], reused=reused, seconds=perf_counter()-started))
    print('Bootstrap %s arc %d complete (%.2fs)' %
          ('reuse' if reused else 'integrate',arc,perf_counter()-started),flush=True)

    return arc,initial,acc


def bootstrap_monthly_scale(settings, arcs, output, workers, cache=None):
    from .bootstrap_cache import validate_bootstrap_settings, input_fingerprints
    arcs = list(arcs)
    if not arcs or len(arcs) != len(set(arcs)):
        raise ValueError('bootstrap requires unique nonempty arcs')
    if cache is not None:
        validate_bootstrap_settings(settings, Path(cache))
    inputs = input_fingerprints(settings, arcs)
    root = Path(output)
    root.mkdir(parents=True,exist_ok=False)
    settings.save(root/'settings')
    write_json(root/'manifest.json',dict(arcs=arcs,scale_scope=settings.span,inputs=inputs,
               cache=str(cache) if cache is not None else None))
    common = np.asarray([settings.raw['DesignAccelerometerConfig_'+sat]['cali_value']['Scale']
                         for sat in ('A','B')],float)
    sh = settings.raw['GravityDesignParameterConfig']['StokesCoefficientsConfig']
    gravity = np.zeros((sh['MaxDegree']+1)**2-sh['MinDegree']**2)
    print('Monthly scale bootstrap: fresh orbit/variational integration, %d arcs, %d workers'
          %(len(arcs),workers),flush=True)
    with h5py.File(root/'state.h5','x') as h:
        h.attrs['scale_scope'] = 'one matrix per satellite over configured processing span'
        h.create_dataset('monthly_scale',data=common)
        h.create_dataset('gravity_delta',data=gravity)
        for number,(arc,initial,acc) in enumerate(map_arcs(
                _bootstrap_arc_job,arcs,(root,cache),workers,ordered=False),1):
            validate_shared_scale(acc,common)
            group = h.create_group('arc_%d'%arc)
            group.create_dataset('initial',data=initial)
            for i,sat in enumerate(('A','B')):
                group.create_dataset('acc_'+sat,data=acc[i])
            h.flush()
            print('Monthly scale bootstrap: %d/%d arcs' % (number,len(arcs)),flush=True)
    if input_fingerprints(settings, arcs) != inputs:
        raise ValueError('bootstrap inputs changed during execution; refusing completion')
    write_json(root/'complete.json',dict(status='complete',arcs=list(arcs),
               scale_scope=list(settings.span),monthly_scale=common.tolist()))
    return root
