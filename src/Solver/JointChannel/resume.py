"""Resume a completed two-iteration joint solve at its interrupted feedback stage."""
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .channel import Settings, write_json
from .feedback import run_feedback
from .workflow import joint_options, load_result, publish_products


def inspect_resume(run_root, setting_directory):
    root = Path(run_root).resolve()
    config_dir = Path(setting_directory).resolve()
    if (root / 'complete.json').exists():
        raise ValueError('joint run already has a final complete.json')
    with open(config_dir / 'CalibrateOrbitConfig.json', encoding='utf-8') as f:
        calibration = json.load(f)
    options = joint_options(calibration['JointEstimationConfig'],
                            calibration['ParallelControlConfig'])
    gls = calibration['GLSIterationConfig']
    if not (options['Enabled'] and options['FeedbackEnabled']
            and options['MaxOuterIterations'] == 2
            and calibration['StepControlConfig']['isGLS']
            and int(gls['MaxIteration']) == 1):
        raise ValueError('resume requires two outer iterations and one GLS iteration')
    first = root / 'outer_00_gls_00'
    second = root / 'outer_01_gls_00'
    first_state = root / 'state_01'
    for directory in (root / 'outer_00_ols', first, first_state,
                      root / 'outer_01_ols', second):
        if not (directory / 'complete.json').is_file():
            raise ValueError('required checkpoint is incomplete: ' + str(directory))
    settings = Settings(second / 'settings', Path.cwd())
    active = Settings(config_dir, Path.cwd())
    active.raw['SolverConfig']['OrbitCovarianceMode'] = gls['OrbitCovarianceMode']
    changed = [name for name in settings.raw if settings.raw[name] != active.raw[name]]
    if changed:
        raise ValueError('current science settings differ from solved checkpoint: ' + ', '.join(changed))
    first_manifest, first_delta = load_result(first)
    second_manifest, second_delta = load_result(second)
    if first_manifest['arcs'] != second_manifest['arcs'] or not second_manifest['arcs']:
        raise ValueError('completed solves have different arc lists')
    if Path(second_manifest['base_state']).resolve() != (first_state / 'state.h5').resolve():
        raise ValueError('last solve did not use the first feedback state')
    if Path(second_manifest['dm_root']).resolve() != (first_state / 'DesignMatrix').resolve():
        raise ValueError('last solve did not use the first feedback design matrices')
    skip = 18 if settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']['isScale'] else 0
    with h5py.File(first_state / 'state.h5', 'r') as h:
        if Path(h.attrs['source_solution']).resolve() != first.resolve():
            raise ValueError('first feedback came from a different solution')
        if not np.isclose(float(h.attrs['damping']), options['Damping']):
            raise ValueError('first feedback damping differs from current configuration')
        if not np.allclose(h['gravity_delta'][()], options['Damping'] * first_delta[skip:],
                           rtol=1e-12, atol=1e-20):
            raise ValueError('first feedback gravity differs from solved increment')
    if first_delta.shape != second_delta.shape:
        raise ValueError('global parameter layouts differ between outer solves')
    output = root / 'state_02_parallel_resume'
    if output.exists():
        raise FileExistsError('resume output already exists: ' + str(output))
    interface = settings.raw['InterfaceConfig']['PathOfFilesConfig']
    tag = '_'.join(settings.span)
    arc_filter = Path(interface['report_arcft']) / (tag + '.txt')
    arc_definition = Path(interface['report_arc']) / (tag + '.txt')
    if not arc_filter.is_file() or not arc_definition.is_file():
        raise FileNotFoundError('arc selection reports are missing')
    return dict(root=root, solution=second, output=output, settings=settings,
                options=options, first_delta=first_delta, second_delta=second_delta,
                arc_filter=arc_filter, arc_definition=arc_definition,
                arc_count=len(second_manifest['arcs']), setting_directory=config_dir)


def resume_joint(run_root, setting_directory, check_only=False):
    plan = inspect_resume(run_root, setting_directory)
    print('Resume from completed second GLS:', plan['solution'], flush=True)
    print('Skip both OLS/GLS rounds and first feedback; remaining feedback: %d arcs with %d workers'
          % (plan['arc_count'], plan['options']['FeedbackProcesses']), flush=True)
    if check_only:
        return plan
    os.chdir(Path(__file__).resolve().parents[3] / 'demo')
    feedback = run_feedback(plan['solution'], plan['output'],
                            plan['options']['Damping'],
                            workers=plan['options']['FeedbackProcesses'])
    base_global = plan['options']['Damping'] * plan['first_delta']
    cumulative = base_global + plan['options']['Damping'] * plan['second_delta']
    products = publish_products(
        plan['settings'], plan['solution'], cumulative, base_global,
        feedback, plan['setting_directory'], plan['arc_filter'],
        plan['arc_definition'], workers=plan['options']['ArcProcesses'])
    record = dict(products=products, resumed_from=str(plan['solution']),
                  previous_feedback=str(plan['root'] / 'state_02'),
                  final_feedback=str(feedback), arcs=plan['arc_count'],
                  workers=dict(ArcProcesses=plan['options']['ArcProcesses'],
                               FeedbackProcesses=plan['options']['FeedbackProcesses']))
    write_json(plan['root'] / 'complete.json', record)
    print('Joint estimation resumed products:', products, flush=True)
    return products
