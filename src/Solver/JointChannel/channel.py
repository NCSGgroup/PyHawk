"""Read legacy products, write only into a fresh joint-channel directory."""
import copy
from datetime import datetime, timezone
from tqdm import tqdm
import importlib
import json
from pathlib import Path
import h5py
import numpy as np
from .core import ObservationBlock, reduce_arc, solve_global, postfit
from .parallel import map_arcs

CONFIG_TYPES = {
    'InterfaceConfig': ('Pre_Interface', 'InterfaceConfig'),
    'SolverConfig': ('Pre_Solver', 'SolverConfig'),
    'NEQConfig': ('Pre_NEQ', 'NEQConfig'),
    'AdjustOrbitConfig': ('Pre_AdjustOrbit', 'AdjustOrbitConfig'),
    'GravityDesignParameterConfig': ('Pre_Parameterization', 'ParameterConfig'),
    'OrbitFitParameterConfig': ('Pre_Parameterization', 'ParameterConfig'),
    'ForceModelConfig': ('Pre_ForceModel', 'ForceModelConfig'),
    'ODEConfig': ('Pre_ODE', 'ODEConfig'),
    'FrameConfig': ('Pre_Frame', 'FrameConfig'),
    'DesignAccelerometerConfig_A': ('Pre_Accelerometer', 'AccelerometerConfig'),
    'DesignAccelerometerConfig_B': ('Pre_Accelerometer', 'AccelerometerConfig'),
    'AccelerometerConfig_A': ('Pre_Accelerometer', 'AccelerometerConfig'),
    'AccelerometerConfig_B': ('Pre_Accelerometer', 'AccelerometerConfig'),
}


class Settings:
    def __init__(self, directory, path_base):
        self.raw = {}
        for name in CONFIG_TYPES:
            with open(Path(directory) / (name + '.json'), encoding='utf-8-sig') as f:
                data = json.load(f)
            # Legacy JSON paths are relative to demo/, NOT the settings directory.
            def resolve(value):
                if isinstance(value, dict):
                    return {k: resolve(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [resolve(v) for v in value]
                if isinstance(value, str) and value.startswith(('./', '../')):
                    return str((Path(path_base) / value).resolve())
                return value
            self.raw[name] = resolve(data)

    def obj(self, name):
        module, cls = CONFIG_TYPES[name]
        obj = getattr(importlib.import_module('src.Preference.' + module), cls)()
        obj.__dict__.update(copy.deepcopy(self.raw[name]))
        return obj

    def save(self, directory):
        Path(directory).mkdir()
        for name, data in self.raw.items():
            write_json(Path(directory) / (name + '.json'), data)

    @property
    def span(self):
        return self.raw['InterfaceConfig']['date_span']

    @property
    def tag(self):
        return '_'.join(self.span)


def write_json(path, data):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def epoch_indices(source, target):
    """Exact microsecond keys; never truncate fractional epochs to seconds."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('non-finite epoch')
    keys = np.rint(source * 1e6).astype(np.int64)
    if len(np.unique(keys)) != len(keys):
        raise ValueError('duplicate source epochs')
    lookup = {int(t): i for i, t in enumerate(keys)}
    try:
        return np.array([lookup[int(t)] for t in np.rint(target * 1e6)], dtype=int)
    except KeyError as exc:
        raise ValueError('observation epoch absent from nominal orbit/raw data: %s' % exc)


def range_rate(r, v):
    dr, dv = r[:, :3] - r[:, 3:], v[:, :3] - v[:, 3:]
    distance = np.linalg.norm(dr, axis=1)
    if np.any(distance <= 0):
        raise ValueError('zero satellite separation')
    return np.einsum('ij,ij->i', dr / distance[:, None], dv)


def read_qc_basis(filename, empirical, apply_cached_mask=True):
    # Deliberately independent of OneCPROutlierOnly: QC role does not disable E.
    with h5py.File(filename, 'r') as h:
        t = h['empirical_prefit_t'][()]
        keep = h['arc_qc_epoch_keep'][()].astype(bool) if apply_cached_mask and 'arc_qc_epoch_keep' in h else np.ones(t.size, bool)
        if keep.shape != t.shape:
            raise ValueError('QC epoch mask mismatch')
        if empirical:
            e = h['design_matrix'][()]
            if e.ndim != 2 or e.shape[0] != t.size:
                raise ValueError('empirical design/QC epoch mismatch')
            e = e[keep]
            # Entirely removed segments have zero support, so no unknown is added.
            columns = np.flatnonzero(np.any(e != 0, axis=0))
            e = e[:, columns]
        else:
            e, columns = np.empty((keep.sum(), 0)), np.empty(0, int)
    return t[keep], e, columns



def read_solved_basis(filename, times, layout, empirical):
    """Recover the exact solved rows/columns, never reapply a cached QC mask."""
    times = np.asarray(times, float)
    columns = np.asarray(layout['empirical_source_columns'], dtype=int)
    count = int(layout['empirical_count'])
    if columns.ndim != 1 or columns.size != count or len(np.unique(columns)) != count:
        raise ValueError('invalid solved empirical column layout')
    if not empirical and count:
        raise ValueError('empirical layout conflicts with solution manifest')
    with h5py.File(filename, 'r') as h:
        rows = epoch_indices(h['empirical_prefit_t'][()], times)
        if not empirical:
            return np.empty((times.size, 0))
        basis = h['design_matrix'][()]
        if basis.ndim != 2 or basis.shape[0] != h['empirical_prefit_t'].size:
            raise ValueError('empirical design/QC epoch mismatch')
        if np.any(columns < 0) or np.any(columns >= basis.shape[1]):
            raise ValueError('solved empirical column absent from source basis')
        selected = basis[rows]
        active = np.flatnonzero(np.any(selected != 0, axis=0))
        if not np.array_equal(active, columns):
            raise ValueError('empirical basis support changed since joint solve')
        selected = selected[:, columns]
        if not np.isfinite(selected).all():
            raise ValueError('non-finite solved empirical basis')
        return selected


def read_arc(settings, arc, empirical=True, covariance=False, dm_root=None):
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload, SatID
    from src.Solver.CovarianceModel import (
        load_covariance, orbit_covariance_filename, range_rate_covariance_filename,
        combine_itsg_epoch_covariance, covariance_matrix_epoch_index,
    )
    interface = settings.obj('InterfaceConfig')
    if interface.sat != {'A': True, 'B': True}:
        raise ValueError('joint channel requires both satellites A and B')
    rd = ArcData(interface)
    solver, neq = settings.raw['SolverConfig'], settings.raw['NEQConfig']
    parameter = settings.raw['GravityDesignParameterConfig']
    if not parameter['TransitionMatrixConfig']['isRequired'] or parameter['TransitionMatrixConfig']['AllisGlobal']:
        raise ValueError('joint channel requires local initial states')
    if parameter['AccelerometerConfig']['AllisGlobal']:
        raise ValueError('all-global ACC layout is unsupported; scale has a separate explicit layout')
    root = Path(dm_root or solver['DesignMatrixTemp']) / settings.tag
    orbit_file, kbr_file = root / ('Orbit_%d_AB.hdf5' % arc), root / ('RangeRate_%d.hdf5' % arc)
    with h5py.File(orbit_file, 'r') as h:
        orbit = {key: h[key][()] for key in ('t', 'r', 'v', 'local_r', 'global_r')}
    with h5py.File(kbr_file, 'r') as h:
        kt, ka, kb = (h[key][()] for key in ('t', 'global', 'local'))
    nl, ng = orbit['local_r'].shape[-1], ka.shape[1]
    if kb.shape[1] != 2 * nl:
        raise ValueError('KBR local layout must be [satellite A, satellite B]')
    scale_global = bool(parameter['AccelerometerConfig']['isScale'])
    expected = int(parameter['StokesCoefficientsConfig']['Parameter_Number']) + (18 if scale_global else 0)
    if ng != expected or orbit['global_r'].shape[-1] != ng - (9 if scale_global else 0):
        raise ValueError('global layout inconsistent with parameter configuration')
    if not neq['SSTConfig']['IsRequired'] or neq['SSTConfig']['measurement'] != 2:
        raise ValueError('joint channel currently requires KBR range rate')
    qcfile = Path(settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['RangeRateTemp']) / settings.tag / ('RangeRate_%d.hdf5' % arc)
    qc_config = settings.raw['AdjustOrbitConfig']['RangeRateConfig']
    physical_qc = qc_config.get('PhysicalOCOutlierEnabled', False)
    if not isinstance(physical_qc, bool):
        raise ValueError('PhysicalOCOutlierEnabled must be a JSON boolean')
    t, e, empirical_columns = read_qc_basis(qcfile, empirical or physical_qc,
                                            apply_cached_mask=not physical_qc)
    qc_audit = None
    if physical_qc:
        from .physical_qc import screen_physical_oc
        raw = rd.getData(arc=arc, kind=Payload.KBR)
        oi = epoch_indices(orbit['t'], t)
        physical_oc = raw[epoch_indices(raw[:, 0], t), 2] - range_rate(orbit['r'][oi], orbit['v'][oi])
        keep = screen_physical_oc(t, physical_oc, e, qc_config)
        qc_audit = dict(source='current_physical_oc', input_count=int(t.size),
                        retained_count=int(keep.sum()), rejected_time=t[~keep].tolist(),
                        empirical_subtracted=False)
        t, e = t[keep], e[keep]
        active = np.any(e != 0, axis=0)
        e, empirical_columns = e[:, active], empirical_columns[active]
        if not empirical:
            e, empirical_columns = np.empty((t.size, 0)), np.empty(0, int)
    if not t.size:
        raise ValueError('arc %s has no retained KBR epochs' % arc)
    total_local = 2 * nl + e.shape[1]
    blocks = []
    mode = solver.get('OrbitCovarianceMode', 'EstimatedFull')
    for i, sat in enumerate((SatID.A, SatID.B)):
        cfg = neq['Orbit%sConfig' % sat.name]
        if not cfg['IsRequired']:
            continue
        if cfg.get('FixKinematicOrbit', False):
            raise ValueError('FixKinematicOrbit is not supported in joint channel')
        payload = Payload.KinematicOrbit if cfg['UseKinematicOrbit'] else Payload.GNV
        observations = rd.getData(arc=arc, kind=payload, sat=sat)
        from src.Solver.OrbitObservations import align_positions
        selected,idx=align_positions(observations,orbit['t'],float(solver.get('OrbitSamplingSeconds',60.)),
                                     solver.get('OrbitSamplingOrigin','GPS'))
        tt = observations[selected, 0]
        a0 = orbit['global_r'][idx, 3*i:3*i+3].reshape(-1, orbit['global_r'].shape[-1])
        a = np.zeros((a0.shape[0], ng))
        if scale_global:
            a[:, 9*i:9*i+9], a[:, 18:] = a0[:, :9], a0[:, 9:]
        else:
            a[:] = a0
        b = np.zeros((a.shape[0], total_local))
        b[:, nl*i:nl*(i+1)] = orbit['local_r'][idx, 3*i:3*i+3].reshape(-1, nl)
        y = (observations[selected, 1:4] - orbit['r'][idx, 3*i:3*i+3]).reshape(-1)
        cov = None
        from .initial_weights import initial_mode, normalized_epoch_covariance, wls_basis, physical_covariance
        if not covariance and initial_mode(solver) == 'KINRelative':
            if payload != Payload.KinematicOrbit or observations.shape[1] < 10:
                raise ValueError('KINRelative requires kinematic XYZ epoch covariance')
            cov = normalized_epoch_covariance(observations[selected, 4:10], solver)
        if covariance:
            mapping = 'PhysicalTime' if mode == 'ITSGEpoch' else solver.get('CovarianceEpochMapping', 'LegacyIndex')
            filename=solver.get('CovarianceFiles',{}).get('Orbit_'+sat.name,orbit_covariance_filename(solver['PostOrbit'], settings.span, sat.name))
            cov = load_covariance(filename,
                                  tt, components=3, matrix_epoch_index=covariance_matrix_epoch_index(mapping, selected))
            if wls_basis(solver):
                cov = physical_covariance(cov, 'Orbit_'+sat.name, solver, observations[selected, 4:10])
            if mode == 'ITSGEpoch':
                if payload != Payload.KinematicOrbit or observations.shape[1] < 10:
                    raise ValueError('ITSGEpoch requires kinematic XYZ covariance')
                cov = combine_itsg_epoch_covariance(cov, observations[selected, 4:10])
        from .arc_weights import precision_factor
        blocks.append(ObservationBlock('Orbit_' + sat.name, a, b, y, cov,
                      weight=precision_factor(solver,arc,'Orbit_'+sat.name),time=tt))
    idx, oi = epoch_indices(kt, t), epoch_indices(orbit['t'], t)
    raw = rd.getData(arc=arc, kind=Payload.KBR)
    y = raw[epoch_indices(raw[:, 0], t), 2] - range_rate(orbit['r'][oi], orbit['v'][oi])
    b = np.column_stack((kb[idx], e))
    filename=solver.get('CovarianceFiles',{}).get('RangeRate',range_rate_covariance_filename(solver['PostKBRR'], settings.span))
    cov = load_covariance(filename, t, components=1) if covariance else None
    if covariance and wls_basis(solver):
        cov = physical_covariance(cov, 'RangeRate', solver)
    from .arc_weights import precision_factor
    weight = (1.0 if covariance else float(solver['OrbitKinFactor']))*precision_factor(solver,arc,'RangeRate')
    blocks.append(ObservationBlock('RangeRate', ka[idx], b, y, cov, weight, t))
    layout = {'local_per_satellite': nl, 'empirical_count': e.shape[1],
              'empirical_source_columns': empirical_columns.tolist(), 'scale_global': scale_global,
              'order': ['A_initial_ACC', 'B_initial_ACC', 'KBR_empirical'],
              'empirical_convention': 'absolute coefficients against raw physical O-C'}
    if qc_audit is not None:
        layout['physical_oc_qc'] = qc_audit
    return blocks, layout


def _reduce_arc_job(arc, settings_directory, output, empirical, covariance,
                    dm_root, rcond):
    settings = Settings(settings_directory, output)
    blocks, layout = read_arc(settings, arc, empirical, covariance, dm_root)
    reduced = reduce_arc(blocks, rcond)
    with h5py.File(Path(output) / ('arc_%d.h5' % arc), 'x') as h:
        h.attrs['layout'] = json.dumps(layout)
        h.attrs['local_rank'] = reduced.local_rank
        h.create_dataset('local_offset', data=reduced.local_offset)
        h.create_dataset('local_global', data=reduced.local_global)
    return reduced.normal, reduced.rhs


def _backsub_arc_job(arc, settings_directory, output, empirical, covariance,
                     dm_root, delta):
    settings = Settings(settings_directory, output)
    blocks, _ = read_arc(settings, arc, empirical, covariance, dm_root)
    with h5py.File(Path(output) / ('arc_%d.h5' % arc), 'a') as h:
        local = h['local_offset'][()] - h['local_global'][()] @ delta
        statistics={}
        with h5py.File(Path(output)/'global.h5','r') as global_file:
            has_probes='redundancy_probes' in global_file
        if has_probes:
            from .arc_weights import block_statistics
            with h5py.File(Path(output)/'global.h5','r') as global_file:
                probes=global_file['redundancy_probes'][()]
            statistics=block_statistics(blocks,h['local_global'][()],delta,local,probes)
        h.create_dataset('local_delta', data=local)
        for block in blocks:
            group = h.create_group(block.name)
            residual = postfit(block, delta, local)
            group.create_dataset('time', data=block.time)
            group.create_dataset('prefit', data=block.residual)
            group.create_dataset('linear_postfit', data=residual)
            group.attrs['rms'] = float(np.sqrt(np.mean(residual**2)))
            group.attrs['precision_weight']=block.weight
            for key,value in statistics.get(block.name,{}).items():group.attrs[key]=value
    return arc


def run_joint(settings, arcs, output, empirical=True, covariance=False, dm_root=None,
              base_state=None, rcond=1e-12, allow_rank_deficient=False, workers=1):
    arcs = [int(a) for a in arcs]
    if not arcs or len(set(arcs)) != len(arcs):
        raise ValueError('provide unique nonempty arc IDs')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # never overwrite an existing run
    settings=copy.deepcopy(settings)
    from .initial_weights import initial_mode, prepare_initial_weights
    if not covariance:
        prepare_initial_weights(settings, arcs, dm_root, workers)
    # A previous GLS stage must keep the exact covariance used in its solve.
    if covariance:
        import shutil
        from src.Solver.CovarianceModel import orbit_covariance_filename,range_rate_covariance_filename
        solver=settings.raw['SolverConfig'];folder=output/'covariance';folder.mkdir()
        files={}
        for name in ('Orbit_A','Orbit_B','RangeRate'):
            if name!='RangeRate' and not settings.raw['NEQConfig']['Orbit'+name[-1]+'Config']['IsRequired']:continue
            source=solver.get('CovarianceFiles',{}).get(name,
                range_rate_covariance_filename(solver['PostKBRR'],settings.span) if name=='RangeRate'
                else orbit_covariance_filename(solver['PostOrbit'],settings.span,name[-1]))
            destination=folder/(name+'.h5');shutil.copy2(source,destination);files[name]=str(destination)
        solver['CovarianceFiles']=files
    settings.save(output / 'settings')
    manifest = {'schema': 'pyhawk-joint-v1', 'arcs': arcs, 'empirical': empirical,
                'covariance': covariance, 'dm_root': str(Path(dm_root or settings.raw['SolverConfig']['DesignMatrixTemp']).resolve()),
                'base_state': str(Path(base_state).resolve()) if base_state else None,
                'rcond': rcond, 'status': 'building', 'date_span': settings.span,
                'workers': int(workers),
                'initial_weighting': settings.raw['SolverConfig'].get('InitialWeightingAudit') if not covariance else None}
    write_json(output / 'manifest.json', manifest)
    n, rhs = None, None
    phase = 'GLS' if covariance else ('Initial WLS' if initial_mode(settings.raw['SolverConfig'])=='KINRelative' else 'OLS')
    print('%s build NEQ: %d arcs, %d workers -> %s' % (phase, len(arcs), workers, output), flush=True)
    arguments = (output / 'settings', output, empirical, covariance, dm_root, rcond)
    for number, (arc_normal, arc_rhs) in enumerate(
            tqdm(map_arcs(_reduce_arc_job, arcs, arguments, workers),
                 total=len(arcs), desc=phase+' Build NEQ', unit='arc'), 1):
        if n is None:
            n, rhs = arc_normal.copy(), arc_rhs.copy()
        else:
            n += arc_normal
            rhs += arc_rhs
        if number % 10 == 0 or number == len(arcs):
            print('Joint reduce: %d/%d arcs' % (number, len(arcs)), flush=True)
    # Persist the freshly rebuilt normal equations before starting the solve.
    # Each run uses an exclusive new directory; no previous OLS/GLS NEQ is reused.
    with h5py.File(output / 'global.h5', 'x') as h:
        h.create_dataset('normal', data=n)
        h.create_dataset('rhs', data=rhs)
    write_json(output / 'neq_complete.json', {
        'status': 'complete', 'phase': phase, 'arcs': len(arcs),
        'covariance': covariance, 'parameters': len(rhs),
        'completed_utc': datetime.now(timezone.utc).isoformat()})
    print('%s build NEQ complete; global solve: %d parameters' % (phase, len(rhs)), flush=True)
    with tqdm(total=1, desc=phase+' Global solve', unit='solve') as progress:
        delta, diagnostic = solve_global(n, rhs, rcond, allow_rank_deficient)
        with h5py.File(output / 'global.h5', 'a') as h:
            h.create_dataset('delta', data=delta)
            from .arc_weights import weight_options,covariance_probes
            weight_config=weight_options(settings.raw['SolverConfig'])
            if covariance and weight_config['Mode']=='VCE':
                h.create_dataset('redundancy_probes',data=covariance_probes(n,
                    weight_config['RedundancyProbes'],weight_config['RandomSeed']))
                h.attrs['redundancy_probe_seed']=weight_config['RandomSeed']
        progress.update(1)
    del n, rhs
    print('Joint back-substitution: %d arcs, %d workers' % (len(arcs), workers), flush=True)
    arguments = (output / 'settings', output, empirical, covariance, dm_root, delta)
    for number, _ in enumerate(tqdm(map_arcs(_backsub_arc_job, arcs, arguments, workers),
                                    total=len(arcs), desc=phase+' Postfit residuals', unit='arc'), 1):
        if number % 10 == 0 or number == len(arcs):
            print('Joint back-substitution: %d/%d arcs' % (number, len(arcs)), flush=True)
    diagnostic['status'] = 'complete'
    write_json(output / 'complete.json', diagnostic)
    return output
