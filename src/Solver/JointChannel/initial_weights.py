"""KIN covariance shape with one span-wide mean precision normalization."""
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from .parallel import map_arcs
from src.Solver.OrbitObservations import align_positions, epoch_covariance


def initial_mode(solver):
    mode = solver.get('InitialOrbitWeighting', 'Unit')
    if mode not in ('Unit', 'KINRelative'):
        raise ValueError('InitialOrbitWeighting must be Unit or KINRelative')
    return mode


def _precision_job(arc, settings, dm_root):
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload, SatID
    solver = settings.raw['SolverConfig']
    path = Path(dm_root or solver['DesignMatrixTemp']) / settings.tag / ('Orbit_%d_AB.hdf5' % arc)
    time = read_epoch_times(path, arc)
    rd = ArcData(settings.obj('InterfaceConfig'))
    trace, count = 0., 0
    for sat in (SatID.A, SatID.B):
        config = settings.raw['NEQConfig']['Orbit' + sat.name + 'Config']
        if not config['IsRequired']:
            continue
        if not config['UseKinematicOrbit']:
            raise ValueError('KINRelative requires KIN observations for every active orbit block')
        obs = rd.getData(arc=arc, kind=Payload.KinematicOrbit, sat=sat)
        rows, _ = align_positions(obs, time, float(solver.get('OrbitSamplingSeconds', 60.)),
                                  solver.get('OrbitSamplingOrigin', 'GPS'))
        if obs.shape[1] < 10:
            raise ValueError('KINRelative requires six epoch covariance columns')
        cov = epoch_covariance(obs[rows, 4:10])
        inv = np.linalg.solve(cov, np.broadcast_to(np.eye(3), cov.shape))
        trace += float(np.trace(inv, axis1=1, axis2=2).sum())
        count += 3 * len(rows)
    return trace, count


def prepare_initial_weights(settings, arcs, dm_root, workers):
    solver = settings.raw['SolverConfig']
    if initial_mode(solver) != 'KINRelative':
        return
    trace, count = 0., 0
    for value, size in tqdm(map_arcs(_precision_job, arcs, (settings, dm_root), workers),
                            total=len(arcs), desc='Initial KIN weight normalization', unit='arc'):
        trace += value
        count += size
    q = trace / count if count else np.nan
    if not np.isfinite(q) or q <= 0:
        raise ValueError('invalid common KIN precision normalization')
    solver['InitialOrbitMeanPrecision'] = q
    solver['InitialWeightingAudit'] = dict(
        orbit_components=count, mean_precision=q, orbit_kin_factor=float(solver['OrbitKinFactor']),
        normalization='one mean trace(S^-1)/3 across all selected epochs and both satellites',
        orbit_units='m', range_rate_units='m/s',
        factor_semantics='relative numerical precision, average initial orbit precision = 1')


def normalized_epoch_covariance(components, solver):
    q = float(solver.get('InitialOrbitMeanPrecision', np.nan))
    if not np.isfinite(q) or q <= 0:
        raise ValueError('KINRelative needs a common precision normalization from run_joint')
    blocks = q * epoch_covariance(components)
    # Only 3x3 blocks are nonzero; support block whitening without a dense matrix.
    return blocks


def wls_basis(solver):
    mode = solver.get('GLSResidualSpace', 'Physical')
    if mode not in ('Physical', 'InitialWLS'):
        raise ValueError('GLSResidualSpace must be Physical or InitialWLS')
    if mode == 'InitialWLS' and initial_mode(solver) != 'KINRelative':
        raise ValueError('InitialWLS requires KINRelative initial weighting')
    if mode == 'InitialWLS' and solver.get('OrbitCovarianceMode', 'EstimatedFull') != 'EstimatedFull':
        raise ValueError('InitialWLS residual space requires EstimatedFull covariance')
    if mode == 'InitialWLS' and (not np.isfinite(float(solver['OrbitKinFactor'])) or float(solver['OrbitKinFactor']) <= 0):
        raise ValueError('InitialWLS requires positive finite OrbitKinFactor')
    return mode == 'InitialWLS'


def basis_residual(residual, name, solver, components=None):
    """Fixed initial WLS transform, before adaptive arc precision."""
    if name == 'RangeRate':
        return np.asarray(residual) * np.sqrt(float(solver['OrbitKinFactor']))
    factors = np.linalg.cholesky(normalized_epoch_covariance(components, solver))
    return np.linalg.solve(factors, np.asarray(residual).reshape(-1,3,1))[:,:,0]


def physical_covariance(covariance, name, solver, components=None):
    """Pull WLS-space covariance back: C = W0^-1 T W0^-T.

    Whitening C is equivalent to whitening T after applying W0 to A, B, y.
    Do not normalize T: its estimated amplitude is part of the GLS model.
    """
    if name == 'RangeRate':
        return np.asarray(covariance) / float(solver['OrbitKinFactor'])
    factors = np.linalg.cholesky(normalized_epoch_covariance(components, solver))
    n = len(factors)
    blocks = np.asarray(covariance).reshape(n,3,n,3)
    result = np.einsum('iab,ibjc,jdc->iajd', factors, blocks, factors, optimize=True).reshape(3*n,3*n)
    # Congruence is symmetric analytically; remove roundoff after contraction.
    return 0.5 * (result + result.T)


def read_epoch_times(path, arc, attempts=3):
    """Retry only explicit filesystem I/O failures, including HDF5 close errors."""
    import time
    for attempt in range(attempts):
        try:
            with h5py.File(path, 'r') as h:
                epochs = h['t'][()]
            return epochs
        except (OSError, RuntimeError) as exc:
            transient = getattr(exc, 'errno', None) == 5 or 'Input/output error' in str(exc) or 'errno = 5' in str(exc)
            if not transient or attempt + 1 == attempts:
                raise RuntimeError('KIN epoch read failed: arc=%s file=%s attempt=%s: %s' % (arc,path,attempt+1,exc)) from exc
            print('Retry KIN epoch I/O: arc=%s file=%s attempt=%s/%s: %s' % (arc,path,attempt+1,attempts,exc),flush=True)
            time.sleep(attempt+1)
