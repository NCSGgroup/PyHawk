"""Independent KBRR selection for each physical linearization."""
from pathlib import Path
import time as clock
import h5py
from .channel import Settings, write_json
from .parallel import map_arcs


def _refresh_arc(arc, settings_dir, dm_root, output):
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload
    from src.Solver.AdjustSST import RangeRate
    settings = Settings(settings_dir, output)
    config = settings.obj('AdjustOrbitConfig')
    options = config.RangeRate()
    options.__dict__.update(config.RangeRateConfig)
    if options.PhysicalOCOutlierEnabled and options.Iterations < 1:
        raise ValueError('physical O-C QC Iterations must be a positive integer')
    raw = ArcData(settings.obj('InterfaceConfig')).getData(arc, Payload.KBR)
    orbit_path = Path(dm_root)/settings.tag/('Orbit_%d_AB.hdf5' % arc)
    for attempt in range(3):
        try:
            with h5py.File(orbit_path, 'r') as h:
                time, r, v = (h[k][()] for k in ('t', 'r', 'v'))
            break
        except (OSError, RuntimeError) as exc:
            if 'Input/output error' not in str(exc):
                raise
            if attempt == 2:
                raise RuntimeError('KBRR QC orbit read failed for arc %d: %s' % (arc, orbit_path)) from exc
            clock.sleep(2 ** attempt)
    filename = Path(output)/settings.tag/('RangeRate_%d.hdf5' % arc)
    RangeRate.from_integrated_orbit(arc, settings.span, options, raw, time, r, v, filename)
    with h5py.File(filename, 'r+') as h:
        h.attrs['qc_source'] = 'raw_KBR_minus_current_integrated_orbit'
        h.attrs['previous_selection_applied'] = False
    return arc


def refresh_kbrr_qc(settings, candidates, previous, dm_root, output, workers):
    """Fresh files and masks; never overwrite a preceding solution's QC basis."""
    from src.Solver.CalibrateOrbit import CalibrateOrbit
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings.save(output/'settings')
    for _ in map_arcs(_refresh_arc, candidates,
                      (output/'settings', dm_root, output), workers):
        pass
    # Reuse the production arc/epoch detector with a fresh, stage-local input.
    detector = CalibrateOrbit()
    detector._CalibrateOrbit__adjust_config = settings.obj('AdjustOrbitConfig')
    detector._CalibrateOrbit__date_span = settings.span
    accepted, rejected = detector._CalibrateOrbit__screenKBRRArcs(
        candidates, res_dir=output/settings.tag)
    write_json(output/'selection.json', dict(
        source='raw_KBR_minus_current_integrated_orbit',
        candidate_arcs=list(candidates), accepted_arcs=accepted, rejected_arcs=rejected,
        reaccepted_arcs=sorted(set(accepted)-set(previous)),
        previous_selection_applied=False))
    if not accepted:
        raise ValueError('fresh KBRR QC rejected every candidate arc')
    settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['RangeRateTemp'] = str(output.resolve())
    settings.raw['SolverConfig']['KBRRQCRebuilt'] = True
    return accepted
