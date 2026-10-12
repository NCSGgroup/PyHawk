"""Per-arc physical feedback workers with a single-writer state file."""
import json
import os
from contextlib import redirect_stderr
from pathlib import Path

import h5py
import numpy as np

from .channel import Settings, epoch_indices, range_rate, read_solved_basis, write_json
from .feedback import integrate_state, update_physical_state
from .parallel import map_arcs


def _feedback_arc_job(arc, source, root, damping, cache=None):
    from src.Interface.Accelerometer import AccCaliPar_V3, Accelerometer_V3
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload

    source, root = Path(source), Path(root)
    with open(source / 'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    settings = Settings(source / 'settings', source)
    p = settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']
    rd = ArcData(settings.obj('InterfaceConfig'))
    with h5py.File(source / 'global.h5', 'r') as h:
        global_delta = h['delta'][()]
    nscale = 18 if p['isScale'] else 0
    base_state = manifest['base_state']
    if base_state:
        with h5py.File(base_state, 'r') as h:
            previous_gravity = h['gravity_delta'][()]
            initial = h['arc_%d/initial' % arc][()]
            acc = [h['arc_%d/acc_%s' % (arc, sat)][()] for sat in ('A', 'B')]
            if p['isScale']:
                from .monthly_scale import validate_shared_scale
                common_scale = h['monthly_scale'][()]
                validate_shared_scale(acc,common_scale)
    else:
        previous_gravity = np.zeros(global_delta.size - nscale)
        path = (Path(settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['OrbitAdjustResTemp'])
                / settings.tag / ('%d_AB.hdf5' % arc))
        with h5py.File(path, 'r') as h:
            initial = h['TransitionMatrix'][()]
            acc = list(h['Accelerometer'][()])
        gnv = rd.getData(arc, Payload.GNV)
        with h5py.File(Path(manifest['dm_root']) / settings.tag / ('Orbit_%d_AB.hdf5' % arc), 'r') as h:
            idx = epoch_indices(h['t'][()], [gnv[4, 0]])[0]
            nominal = np.column_stack((h['r'][idx].reshape(2, 3),
                                       h['v'][idx].reshape(2, 3)))
        if not np.allclose(initial, nominal, rtol=0, atol=1e-5):
            raise ValueError('cached DM and OrbitAdjustment initial state differ; rebuild consistent inputs')
        for i, sat in enumerate(('A', 'B')):
            cfg = settings.obj('DesignAccelerometerConfig_' + sat)
            ap = AccCaliPar_V3(rd, sat, arc, p['X_AdjustLength'],
                               p['Y_AdjustLength'], p['Z_AdjustLength'], cfg)
            if acc[i].size != ap.getPar().size:
                raise ValueError('bootstrap ACC width differs from design mask')
            modeled = Accelerometer_V3(ap).updatePar(acc[i]).get_acceleration()
            filename = (Path(settings.raw['ForceModelConfig']['PathOfFilesConfig']['temp_non_conservative_data'])
                        / (settings.tag + '_%s_%d.hdf5' % (sat, arc)))
            with h5py.File(filename, 'r') as h:
                if h['acc'].shape != modeled.shape or not np.allclose(
                        h['acc'][()], modeled, rtol=1e-10, atol=1e-15):
                    raise ValueError('cached force and ACC parameters differ; cannot bootstrap feedback safely')

    with h5py.File(source / ('arc_%d.h5' % arc), 'r') as solved:
        local_delta = solved['local_delta'][()]
        layout = json.loads(solved.attrs['layout'])
        old_t = solved['RangeRate/time'][()]
        old_y = solved['RangeRate/prefit'][()]
        predicted = solved['RangeRate/linear_postfit'][()]
        orbit_inputs = {}
        for sat in ('A', 'B'):
            key = 'Orbit_' + sat
            if key in solved:
                orbit_inputs[sat] = (solved[key + '/time'][()],
                                     solved[key + '/prefit'][()],
                                     solved[key + '/linear_postfit'][()])

    expected_nl = 6 + len(acc[0]) - (9 if p['isScale'] else 0)
    if layout['local_per_satellite'] != expected_nl or layout['scale_global'] != p['isScale']:
        raise ValueError('joint solution layout differs from physical state')
    initial_new, acc_new, gravity = update_physical_state(
        initial, *acc, local_delta, global_delta, previous_gravity,
        p['isScale'], damping)
    if p['isScale']:
        validate_shared_scale(acc_new,common_scale+damping*global_delta[:18].reshape(2,9))
    reused = None
    if cache is not None:
        from .feedback_cache import reuse_integration
        reused = reuse_integration(settings, arc, Path(cache), root)
    if reused is None:
        print('Feedback integrate arc %d' % arc, flush=True)
        with open(os.devnull, 'w') as quiet, redirect_stderr(quiet):
            t, r, v = integrate_state(settings, arc, initial_new, acc_new, gravity, root)
    else:
        t, r, v = reused
        print('Feedback reuse arc %d' % arc, flush=True)
    idx = epoch_indices(t, old_t)
    raw = rd.getData(arc, Payload.KBR)
    true_prefit = raw[epoch_indices(raw[:, 0], old_t), 2] - range_rate(r[idx], v[idx])
    qc = (Path(settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['RangeRateTemp'])
          / settings.tag / ('RangeRate_%d.hdf5' % arc))
    e = read_solved_basis(qc, old_t, layout, manifest['empirical'])
    empirical = local_delta[2*expected_nl:]
    true_postfit = true_prefit - damping * (e @ empirical)
    linear = old_y + damping * (predicted - old_y)
    orbit = {}
    for i, sat in enumerate(('A', 'B')):
        if sat not in orbit_inputs:
            continue
        ot, old, fitted = orbit_inputs[sat]
        cfg = settings.raw['NEQConfig']['Orbit%sConfig' % sat]
        payload = Payload.KinematicOrbit if cfg['UseKinematicOrbit'] else Payload.GNV
        obs = rd.getData(arc, payload, sat)
        physical = (obs[epoch_indices(obs[:, 0], ot), 1:4]
                    - r[epoch_indices(t, ot), i*3:i*3+3]).reshape(-1)
        expected = old + damping * (fitted - old)
        orbit[sat] = dict(time=ot, physical_prefit=physical,
                          linear_prediction=expected,
                          nonlinear_difference_rms=float(np.sqrt(np.mean((physical-expected)**2))))
    diagnostic = dict(prefit_rms=float(np.sqrt(np.mean(true_prefit**2))),
                      true_postfit_rms=float(np.sqrt(np.mean(true_postfit**2))),
                      nonlinear_difference_rms=float(np.sqrt(np.mean((true_postfit-linear)**2))))
    data = dict(initial=initial_new, acc_A=acc_new[0], acc_B=acc_new[1],
                kbr_time=old_t, kbr_physical_prefit=true_prefit,
                kbr_true_postfit=true_postfit, kbr_linear_prediction=linear,
                empirical_applied=damping*empirical)
    return arc, data, orbit, diagnostic


def run_feedback_parallel(solution, output, damping, workers, cache=None):
    source, root = Path(solution).resolve(), Path(output).resolve()
    if not (source / 'complete.json').is_file():
        raise ValueError('joint solution is incomplete')
    if root.exists():
        raise FileExistsError(root)
    with open(source / 'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    if cache is not None:
        with h5py.File(Path(cache)/'state.h5','r') as saved:
            if Path(saved.attrs['source_solution']).resolve() != source or float(saved.attrs['damping']) != damping:
                raise ValueError('feedback cache source/damping mismatch')
    settings = Settings(source / 'settings', source)
    parameter = settings.raw['GravityDesignParameterConfig']
    p = parameter['AccelerometerConfig']
    if not p['isRequired'] or parameter['TransitionMatrixConfig']['Parameter_Number'] != 6:
        raise ValueError('feedback requires 6 initial-state parameters and ACC estimation')
    base_state = manifest['base_state']
    if base_state and Path(manifest['dm_root']).resolve() != Path(base_state).resolve().parent / 'DesignMatrix':
        raise ValueError('base state and design matrix do not belong to the same linearization')
    if p['isScale'] and not base_state:
        raise ValueError('monthly scale feedback requires the shared bootstrap state')
    if not base_state:
        fit = settings.raw['OrbitFitParameterConfig']['AccelerometerConfig']
        for key in ('X_AdjustLength', 'Y_AdjustLength', 'Z_AdjustLength'):
            if p[key] != fit[key]:
                raise ValueError('bootstrap requires identical fit/design ACC segmentation')
        for sat in ('A', 'B'):
            a, b = settings.raw['AccelerometerConfig_' + sat], settings.raw['DesignAccelerometerConfig_' + sat]
            if a['cali'] != b['cali'] or a['cali_value'] != b['cali_value']:
                raise ValueError('bootstrap requires identical fit/design ACC parameterization')
    with h5py.File(source / 'global.h5', 'r') as h:
        global_delta = h['delta'][()]
    nscale = 18 if p['isScale'] else 0
    previous_gravity = np.zeros(global_delta.size - nscale)
    if base_state:
        with h5py.File(base_state, 'r') as h:
            previous_gravity = h['gravity_delta'][()]
            if p['isScale']:
                common_scale = h['monthly_scale'][()]
    gravity = previous_gravity + damping*global_delta[nscale:]
    root.mkdir(parents=True, exist_ok=False)
    settings.save(root / 'settings')
    diagnostics = {}
    print('Joint feedback integration: %d arcs, %d workers' % (len(manifest['arcs']), workers), flush=True)
    with h5py.File(root / 'state.h5', 'x') as out:
        out.attrs['source_solution'] = str(source)
        out.attrs['damping'] = damping
        out.attrs['gravity_semantics'] = 'cumulative correction to original configured reference'
        out.create_dataset('gravity_delta', data=gravity)
        if p['isScale']:
            out.create_dataset('monthly_scale',data=common_scale+damping*global_delta[:18].reshape(2,9))
        for number, (arc, data, orbit, diagnostic) in enumerate(map_arcs(
                _feedback_arc_job, manifest['arcs'], (source, root, damping, cache), workers), 1):
            group = out.create_group('arc_%d' % arc)
            for name, value in data.items():
                group.create_dataset(name, data=value)
            for sat, values in orbit.items():
                pod = group.create_group('Orbit_' + sat)
                for name in ('time', 'physical_prefit', 'linear_prediction'):
                    pod.create_dataset(name, data=values[name])
                pod.attrs['nonlinear_difference_rms'] = values['nonlinear_difference_rms']
            diagnostics[str(arc)] = diagnostic
            if number % 10 == 0 or number == len(manifest['arcs']):
                print('Joint feedback integration: %d/%d arcs' % (number, len(manifest['arcs'])), flush=True)
    write_json(root / 'complete.json', dict(status='complete', source_solution=str(source),
                                            damping=damping, arcs=manifest['arcs'], diagnostics=diagnostics))
    return root
