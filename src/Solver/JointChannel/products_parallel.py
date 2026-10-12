"""Parallel per-arc product preparation; standard daily files have one writer."""
import json
from tqdm import tqdm
from pathlib import Path

import h5py
import numpy as np

from .channel import Settings, epoch_indices, read_arc
from .core import ObservationBlock
from .parallel import map_arcs


def _residual_arc_job(arc, solution, cumulative, feedback, l1c):
    from src.Auxilary.StandardKBRR import StandardKBRR
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload, SatID
    from .workflow import save_h5

    solution = Path(solution)
    settings = Settings(solution / 'settings', solution)
    with open(solution / 'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    solver = settings.raw['SolverConfig']
    adjust = settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']
    tag = '-'.join(settings.span)
    skip = 18 if settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']['isScale'] else 0
    with h5py.File(solution / ('arc_%d.h5' % arc), 'r') as solved:
        local = solved['local_delta'][()]
        layout = json.loads(solved.attrs['layout'])
        nl = layout['local_per_satellite']
        time = solved['RangeRate/time'][()]
        pre = solved['RangeRate/prefit'][()]
        post = solved['RangeRate/linear_postfit'][()]
        orbits = {}
        for sat in ('A', 'B'):
            key = 'Orbit_' + sat
            if key in solved:
                orbits[sat] = (solved[key+'/time'][()],
                               solved[key+'/linear_postfit'][()])
    damp = 1.0
    if feedback:
        with h5py.File(Path(feedback)/'state.h5', 'r') as state:
            damp = float(state.attrs['damping'])
            post = state['arc_%d/kbr_true_postfit' % arc][()]
            for sat in orbits:
                orbits[sat] = (orbits[sat][0],
                               state['arc_%d/Orbit_%s/physical_prefit' % (arc, sat)][()])
    dmfile = Path(manifest['dm_root'])/settings.tag/('RangeRate_%d.hdf5' % arc)
    with h5py.File(dmfile, 'r') as h:
        indices = epoch_indices(h['t'][()], time)
        geofit = h['global'][()][indices, skip:] @ cumulative[skip:]
    save_h5(Path(solver['PostParameter'])/tag/('%d_AB.hdf5' % arc),
            dict(par_a=damp*local[:nl], par_b=damp*local[nl:2*nl]),
            dict(parameter_semantics='joint local increments, including initial state', joint_channel=True))
    save_h5(Path(solver['PostKBRR'])/tag/('RangeRate_%d.hdf5' % arc),
            dict(time=time, kbrr_par=damp*local, post_kbrr=post, geofit_kbrr=geofit),
            dict(local_order=json.dumps(layout['order']), joint_channel=True,
                 residual_type='reintegrated' if feedback else 'linear',
                 geofit_semantics='final linearization Jacobian times cumulative SH correction'))
    for sat, (ot, residual) in orbits.items():
        save_h5(Path(solver['PostOrbit'])/tag/('Orbit_%d%s.hdf5' % (arc, sat)),
                dict(time=ot, post_Orbit=residual), dict(joint_channel=True))
    if l1c:
        from src.Frame.Frame import Frame, EOP
        frame_config = settings.obj('FrameConfig')
        frame = Frame(EOP().configure(frameConfig=frame_config).load()).configure(frame_config)
        rd = ArcData(settings.obj('InterfaceConfig'))
        a = rd.getData(arc, Payload.GNV, SatID.A)[:, :4]
        b = rd.getData(arc, Payload.GNV, SatID.B)[:, :4]
        lon, lat, alt = StandardKBRR.midpoint_coordinates(time, a, b, frame)
        path = adjust.get('StandardKBRRTemp', StandardKBRR.output_directory(adjust['RangeRateTemp']))
        for stage, value in (('Raw', pre), ('Pre', pre), ('Post', post), ('Geofit', geofit)):
            StandardKBRR().configure(path, settings.span, stage,
                mission=settings.raw['InterfaceConfig']['mission']).save_arc(
                    time, lon, lat, alt, value, arc)
    return arc


def export_residuals_parallel(settings, solution, cumulative, feedback, l1c, workers):
    from src.Auxilary.StandardKBRR import StandardKBRR
    solution = Path(solution)
    with open(solution/'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    writer = None
    if l1c:
        adjust = settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']
        path = adjust.get('StandardKBRRTemp', StandardKBRR.output_directory(adjust['RangeRateTemp']))
        writer = StandardKBRR().configure(path, settings.span, 'Post',
            mission=settings.raw['InterfaceConfig']['mission']).prepare()
    for _ in tqdm(map_arcs(_residual_arc_job, manifest['arcs'],
                      (solution, cumulative, feedback, l1c), workers),
                      total=len(manifest['arcs']), desc='Export postfit residuals', unit='arc'):
        pass
    return writer.finalize() if writer else []


def _acc_arc_job(arc, solution, feedback):
    from src.Auxilary.StandardACC import StandardACC
    from src.Interface.Accelerometer import AccCaliPar_V3, Accelerometer_V3
    from src.Interface.ArcOperation import ArcData
    solution = Path(solution)
    settings = Settings(solution/'settings', solution)
    p = settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']
    rd = ArcData(settings.obj('InterfaceConfig'))
    if feedback:
        with h5py.File(Path(feedback)/'state.h5', 'r') as h:
            parameters = [h['arc_%d/acc_%s' % (arc, sat)][()] for sat in ('A','B')]
    else:
        with open(solution/'manifest.json',encoding='utf-8') as f:
            manifest = json.load(f)
        if manifest['base_state']:
            with h5py.File(manifest['base_state'],'r') as h:
                parameters = [h['arc_%d/acc_%s'%(arc,sat)][()] for sat in ('A','B')]
        else:
            fit = settings.raw['OrbitFitParameterConfig']['AccelerometerConfig']
            for key in ('X_AdjustLength','Y_AdjustLength','Z_AdjustLength'):
                if fit[key] != p[key]:
                    raise ValueError('ACC1C export requires matching fit/design ACC segmentation')
            for sat in ('A','B'):
                fitcfg, designcfg = settings.raw['AccelerometerConfig_'+sat], settings.raw['DesignAccelerometerConfig_'+sat]
                if fitcfg['cali'] != designcfg['cali'] or fitcfg['cali_value'] != designcfg['cali_value']:
                    raise ValueError('ACC1C export requires matching fit/design parameterization')
            initialfile = (Path(settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['OrbitAdjustResTemp'])
                           / settings.tag / ('%d_AB.hdf5'%arc))
            with h5py.File(initialfile,'r') as h:
                parameters = list(h['Accelerometer'][()])
        with h5py.File(solution/('arc_%d.h5'%arc),'r') as h:
            local = h['local_delta'][()]
            nl = json.loads(h.attrs['layout'])['local_per_satellite']
        nscale = 9 if p['isScale'] else 0
        with h5py.File(solution/'global.h5','r') as h:
            delta = h['delta'][()]
        for i in range(2):
            if len(parameters[i]) != nl-6+nscale:
                raise ValueError('ACC1C parameter width mismatch')
            parameters[i][nscale:] += local[i*nl+6:(i+1)*nl]
            if nscale:
                parameters[i][:9] += delta[i*9:(i+1)*9]
    for i,sat in enumerate(('A','B')):
        cfg = settings.obj('DesignAccelerometerConfig_'+sat)
        ap = AccCaliPar_V3(rd,sat,arc,p['X_AdjustLength'],p['Y_AdjustLength'],p['Z_AdjustLength'],cfg)
        ac = Accelerometer_V3(ap).updatePar(parameters[i])
        original = settings.raw['AccelerometerConfig_'+sat]
        path = original.get('StandardACCTemp', StandardACC.output_directory(original['temp_non_conservative_data']))
        StandardACC().configure(path,settings.span,settings.raw['InterfaceConfig']['mission']).save_arc(
            ap.acc[:,0], ac.get_acceleration(), ac.get_bias(), sat, arc,
            scale_matrix=ac.get_scale_matrix(),
            scale_gcrs=ac.get_scale_gcrs_diagonal())
    return arc


def export_acc_parallel(settings, solution, feedback, workers):
    from src.Auxilary.StandardACC import StandardACC
    solution = Path(solution)
    with open(solution/'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    writers = {}
    for sat in ('A','B'):
        original = settings.raw['AccelerometerConfig_'+sat]
        path = original.get('StandardACCTemp', StandardACC.output_directory(original['temp_non_conservative_data']))
        if path not in writers:
            writers[path] = StandardACC().configure(path,settings.span,
                settings.raw['InterfaceConfig']['mission']).prepare(['A','B'])
    for _ in tqdm(map_arcs(_acc_arc_job, manifest['arcs'], (solution, feedback), workers),
                  total=len(manifest['arcs']), desc='Export ACC1C', unit='arc'):
        pass
    return [str(path) for writer in writers.values() for path in writer.finalize()]


def _rss_arc_job(arc, solution):
    solution = Path(solution)
    settings = Settings(solution/'settings', solution)
    with open(solution/'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    blocks, _ = read_arc(settings, arc, manifest['empirical'],
                         manifest['covariance'], manifest['dm_root'])
    rss, count = 0., 0
    with h5py.File(solution/('arc_%d.h5'%arc),'r') as h:
        local_rank = int(h.attrs['local_rank'])
        for block in blocks:
            residual = h[block.name+'/linear_postfit'][()]
            empty = np.empty((residual.size,0))
            _,_,white = ObservationBlock('residual',empty,empty,residual,
                                          block.covariance,block.weight).whiten()
            rss += float(white @ white)
            count += residual.size
    return rss, count, local_rank


def rss_parallel(solution, arcs, workers):
    return map_arcs(_rss_arc_job, arcs, (solution,), workers)
