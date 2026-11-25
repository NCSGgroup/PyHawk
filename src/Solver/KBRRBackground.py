"""Linearized KBRR restoration of the original dated reference minus static field."""
import copy
import numpy as np
from src.Auxilary.SortCS import SortCS
from src.Interface.LoadSH import LoadGOCO06S


def restore_kbrr_background(design_sh, geofit, force_config, sh_config, date_span):
    """Use the same SH Jacobian/epochs as cumulative geofit; estimated degrees only."""
    design = np.asarray(design_sh, dtype=float)
    fitted = np.asarray(geofit, dtype=float)
    nmax, nmin = int(sh_config['MaxDegree']), int(sh_config['MinDegree'])
    order = sh_config['SortMethod']
    count = (nmax + 1)**2 - nmin**2
    if fitted.ndim != 1 or design.ndim != 2 or design.shape != (fitted.size, count):
        raise ValueError('SH design/geofit shape does not match configured degree range')
    if not np.isfinite(design).all() or not np.isfinite(fitted).all():
        raise ValueError('Nonfinite SH design or geofit')
    force = copy.deepcopy(force_config)
    epoch = force.resolve_time_variable_date(date_span)
    reference = force.RefGravModelConfig
    model = reference['StaticModel']
    delta = np.zeros(count)
    path = ''
    if model == 'GOCO06S' and epoch is not None:
        path = force.PathOfFilesConfig['GOCO06s']
        dated = LoadGOCO06S().load(path, date=epoch)
        static = LoadGOCO06S().load(path, date=None)
        if nmax > min(dated.maxDegree, static.maxDegree, int(reference['Nmax'])):
            raise ValueError('Requested restoration exceeds reference field degree')
        cd, sd = dated.getCS(nmax)
        c0, s0 = static.getCS(nmax)
        delta = SortCS(nmax, nmin, order).sort(cd-c0, sd-s0)
    background = design @ delta
    datasets = dict(kbrr_background_timevar=background,
                    geofit_kbrr_static=fitted + background)
    attrs = dict(kbrr_background_units='m/s',
                 kbrr_background_reference_model=model,
                 kbrr_background_reference_file=str(path),
                 kbrr_background_reference_epoch=str(epoch),
                 kbrr_background_min_degree=nmin,
                 kbrr_background_max_degree=nmax,
                 kbrr_background_sort_method=order,
                 kbrr_background_definition='A_SH @ (original dated reference - static reference)',
                 geofit_kbrr_static_definition='geofit_kbrr + kbrr_background_timevar; linearized; estimated degrees only')
    return datasets, attrs
