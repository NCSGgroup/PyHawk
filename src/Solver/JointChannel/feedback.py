"""Stage 2: explicit physical-state feedback, isolated from legacy products."""
import json
import os
from time import perf_counter
from pathlib import Path
import h5py
import numpy as np
from .channel import Settings, write_json, epoch_indices, range_rate


class OffsetGravityModel:
    """Add cumulative SH corrections to the original reference, exactly once."""
    def __init__(self, reference, c, s):
        self.reference = reference
        self.c, self.s = np.asarray(c, float).copy(), np.asarray(s, float).copy()
        if self.c.shape != self.s.shape or not np.isfinite(self.c).all() or not np.isfinite(self.s).all():
            raise ValueError('invalid cumulative gravity correction')

    def getCS(self, degree):
        c, s = self.reference.getCS(degree)
        c, s = np.asarray(c).copy(), np.asarray(s).copy()
        size = min(c.size, self.c.size)
        c[:size] += self.c[:size]
        s[:size] += self.s[:size]
        return c, s

    def __getattr__(self, name):
        return getattr(self.reference, name)


def update_physical_state(initial, acc_a, acc_b, local_delta, global_delta,
                          previous_gravity, scale_global=False, damping=1.0):
    if not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError('damping must be in (0, 1]')
    initial = np.asarray(initial, float).copy()
    acc = [np.asarray(acc_a, float).copy(), np.asarray(acc_b, float).copy()]
    if initial.shape != (2, 6) or acc[0].shape != acc[1].shape:
        raise ValueError('expected (2,6) initial states and equal ACC layouts')
    local = np.asarray(local_delta, float)
    nscale = 9 if scale_global else 0
    nl = 6 + acc[0].size - nscale
    if local.size < 2 * nl:
        raise ValueError('local parameter vector is too short')
    for i in range(2):
        initial[i] += damping * local[i*nl:i*nl+6]
        acc[i][nscale:] += damping * local[i*nl+6:(i+1)*nl]
        if scale_global:
            acc[i][:9] += damping * global_delta[i*9:(i+1)*9]
    dg = np.asarray(global_delta)[2*nscale:]
    if np.shape(previous_gravity) != dg.shape:
        raise ValueError('gravity state shape mismatch')
    gravity = np.asarray(previous_gravity) + damping * dg
    if not all(np.isfinite(a).all() for a in (initial, acc[0], acc[1], gravity)):
        raise ValueError('non-finite physical state')
    return initial, acc, gravity


def integrate_state(settings, arc, initial, acc_parameters, gravity_delta, output_root):
    """Use legacy force/ODE classes with fresh instances and redirected outputs."""
    from src.Interface.Accelerometer import AccCaliPar_V3, Accelerometer_V3
    from src.Interface.ArcOperation import ArcData
    from src.Solver.AdjustDM import AdjustDM
    from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
    from src.ODE.Instance.GravimetryODE import GravimetryODE
    from src.Auxilary.SortCS import SortCS
    from src.Preference.EnumType import ForceType, GravHarType
    started = perf_counter()
    root = Path(output_root)
    parameter = settings.obj('GravityDesignParameterConfig')
    interface = settings.obj('InterfaceConfig')
    rd = ArcData(interface)
    p = parameter.AccelerometerConfig
    force = settings.obj('ForceModelConfig')
    force.resolve_time_variable_date(settings.span)
    force.NonConservativeConfig['date_span'] = settings.span
    force.PathOfFilesConfig['temp_non_conservative_data'] = str(root / 'NonConservativeForce')
    counts = []
    for i, sat in enumerate(('A', 'B')):
        cfg = settings.obj('DesignAccelerometerConfig_' + sat)
        cfg.temp_non_conservative_data = str(root / 'NonConservativeForce')
        cfg.StandardACCTemp = str(root / 'StandardACC')
        ap = AccCaliPar_V3(rd, sat, arc, p['X_AdjustLength'], p['Y_AdjustLength'], p['Z_AdjustLength'], cfg)
        if len(ap.getPar()) != len(acc_parameters[i]):
            raise ValueError('ACC state/design mask mismatch for satellite ' + sat)
        if p['isScale'] and not all(cfg.cali['Scale']):
            raise ValueError('global-scale layout requires all 9 scale parameters')
        Accelerometer_V3(ap).updatePar(acc_parameters[i]).get_and_save()
        counts.append(len(ap.getPar()))
    if counts[0] != counts[1]:
        raise ValueError('feedback currently requires equal ACC widths for A and B')
    solver = settings.obj('SolverConfig')
    solver.DesignMatrixTemp = str(root / 'DesignMatrix')
    (root / 'DesignMatrix' / settings.tag).mkdir(parents=True, exist_ok=True)
    # Existing AdjustDM exposes its derivative and ODE as protected interfaces.
    # Configure reads legacy initial values, but the ODE is replaced BEFORE propagation.
    dm = AdjustDM(settings.span, ['A', 'B']).configure(
        SerDer=Orbit_var_2nd_diff, arcNo=arc, ODEConfig=settings.obj('ODEConfig'),
        AdjustConfig=settings.obj('AdjustOrbitConfig'), FMConfig=force,
        ParameterConfig=parameter, SolverConfig=solver, InterfaceConfig=interface,
        FrameConfig=settings.obj('FrameConfig'))
    grav = dm._SerDer.ForceType[ForceType.GravHar.name]
    reference = grav.GravHarForce[GravHarType.ReferenceModel.name]
    sh = parameter.StokesCoefficientsConfig
    c, s = SortCS(sh['MaxDegree'], sh['MinDegree'], sh['SortMethod']).invert(gravity_delta)
    if force.RefGravModelConfig['Nmax'] < sh['MaxDegree']:
        raise ValueError('reference degree cannot truncate estimated gravity')
    reference.setModel(OffsetGravityModel(reference.getModelPar(), c, s))
    # Width derives from actual ACC masks, avoiding the legacy node/scale double count.
    nodes = counts[0] - (9 if p['isScale'] else 0)
    start = dm._SerDer.getInitData()[0]
    dm.times = nodes
    dm._ODE = (GravimetryODE().configure(settings.obj('ODEConfig'), parameter)
               .setDataTime(nodes).setParNum()
               .setInitial(start, initial[:, :3].copy(), initial[:, 3:].copy())
               .set2ndDerivative(dm._SerDer))
    setup_done = perf_counter()
    try:
        state = dm._ODE.propagate()
        propagated = perf_counter()
        # Keep the existing finite/physical and disk round-trip verification.
        dm._save_StateVectors(state)
        saved = perf_counter()
        result = write_design_matrices(settings, arc, root / 'DesignMatrix', nodes, state=state)
        finished = perf_counter()
        timings = dict(setup=setup_done-started, integration=propagated-setup_done,
                       verified_save=saved-propagated, design=finished-saved,
                       total=finished-started)
        timing_dir = root / 'timings'
        timing_dir.mkdir(exist_ok=True)
        write_json(timing_dir / ('arc_%d.json' % arc), timings)
        print('Dynamics arc %d seconds: setup=%.2f integration=%.2f verified_save=%.2f design=%.2f total=%.2f'
              % (arc, timings['setup'], timings['integration'], timings['verified_save'],
                 timings['design'], timings['total']), flush=True)
        return result
    finally:
        # Legacy PlanetEphemerides has no public close(). Keep this compatibility
        # seam here so repeated new-channel arcs do not leak SPK file handles.
        ep = getattr(dm._SerDer, '_Assemble2ndDerivative__ep', None)
        kernel = getattr(ep, '_PlanetEphemerides__kernel', None)
        if kernel is not None:
            kernel.close()


def write_design_matrices(settings, arc, dm_root, nodes, state=None, block_size=32):
    """Blockwise projection, using the verified in-memory state when available.

    Preserve legacy dataset names and full integration sampling. Publish each
    output atomically; a completion checkpoint is written only after both exist.
    """
    from contextlib import ExitStack
    from src.Solver.Parameterization import Parameterization
    from src.Auxilary.SSTpartial import SSTpartial
    folder = Path(dm_root) / settings.tag
    pp = Parameterization().configure(settings.obj('GravityDesignParameterConfig'), nodes)
    scale = settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']['isScale']
    orbit = folder / ('Orbit_%d_AB.hdf5' % arc)
    rate = folder / ('RangeRate_%d.hdf5' % arc)
    if orbit.exists() or rate.exists():
        raise FileExistsError('design output already exists for arc %d' % arc)
    otmp, ktmp = (p.with_name(p.name + '.writing-' + str(os.getpid())) for p in (orbit, rate))
    with ExitStack() as stack:
        if state is None:
            raw = stack.enter_context(h5py.File(folder / ('%d_AB.hdf5' % arc), 'r'))
            t, rr, vv = raw['t'][()], raw['r'], raw['v']
        else:
            t, rr, vv = state
        n = len(t)
        if n == 0 or block_size < 1:
            raise ValueError('empty state or invalid block size')
        physical_r, physical_v = np.empty((n, 6)), np.empty((n, 6))
        oh = stack.enter_context(h5py.File(otmp, 'x'))
        kh = stack.enter_context(h5py.File(ktmp, 'x'))
        oh.create_dataset('t', data=t)
        kh.create_dataset('t', data=t)
        for start in range(0, n, block_size):
            stop = min(n, start+block_size)
            r, lr, gr = pp.split_global_local(rr[start:stop], True)
            v, lv, gv = pp.split_global_local(vv[start:stop], True)
            physical_r[start:stop], physical_v[start:stop] = r, v
            pra, pva, prb, pvb = SSTpartial(r[:,:3], v[:,:3], r[:,3:], v[:,3:]).getPartial_RangeRate()
            def project(pr, pv, dr, dv):
                return np.einsum('ij,ijk->ik',pr,dr) + np.einsum('ij,ijk->ik',pv,dv)
            ba, bb = project(pra,pva,lr[:,:3],lv[:,:3]), project(prb,pvb,lr[:,3:],lv[:,3:])
            ga, gb = project(pra,pva,gr[:,:3],gv[:,:3]), project(prb,pvb,gr[:,3:],gv[:,3:])
            g = np.column_stack((ga[:,:9],gb[:,:9],ga[:,9:]+gb[:,9:])) if scale else ga+gb
            for h, values in ((oh,dict(r=r,v=v,local_r=lr,global_r=gr,local_v=lv,global_v=gv)),
                              (kh,dict(local=np.column_stack((ba,bb)), **{'global':g}))):
                for key, value in values.items():
                    if start == 0:
                        h.create_dataset(key, shape=(n,)+value.shape[1:], dtype=value.dtype)
                    h[key][start:stop] = value
    os.replace(str(otmp), str(orbit))
    os.replace(str(ktmp), str(rate))
    return np.asarray(t), physical_r, physical_v


def run_feedback(solution, output, damping=1.0, workers=1):
    """Use the process-pool feedback implementation for every worker count."""
    from .feedback_parallel import run_feedback_parallel
    return run_feedback_parallel(solution, output, damping, int(workers))
