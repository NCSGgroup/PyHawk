"""Screen current physical O-C; the empirical fit is never the observation."""
import numpy as np
from scipy.linalg import lstsq
from src.Auxilary.Outlier import Outlier, EpochOutlier


def screen_physical_oc(time, oc, design, config):
    time, oc, design = np.asarray(time), np.asarray(oc), np.asarray(design)
    if design.ndim != 2 or design.shape[0] != oc.size or not design.shape[1]:
        raise ValueError('physical O-C QC requires cached empirical design rows')
    if not np.isfinite(design).all() or not np.isfinite(oc).all():
        raise ValueError('nonfinite physical O-C or empirical design')
    iterations = config.get('Iterations', 5)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError('physical O-C QC Iterations must be a positive integer')
    keep = np.ones(oc.size, dtype=bool)
    # Cached design is block diagonal by empirical interval. Detect each
    # interval separately, so one noisy interval cannot set all thresholds.
    supports = np.any(design != 0, axis=0)
    if not supports.all():
        design = design[:, supports]
    groups = {}
    for row, support in enumerate(design != 0):
        # Polynomial/sine columns may vanish at interval boundaries. Columns
        # active anywhere in the same connected support belong to one segment.
        active = set(np.flatnonzero(support).tolist())
        if not active:
            raise ValueError('physical O-C QC epoch has no empirical support')
        groups[row] = active
    segments = []
    for row, active in groups.items():
        hits = [i for i, (_, cols) in enumerate(segments) if cols & active]
        rows, cols = [row], active
        for i in reversed(hits):
            rr, cc = segments.pop(i); rows.extend(rr); cols |= cc
        segments.append((rows, cols))
    for rows, cols in segments:
        rows = np.array(sorted(rows)); cols = np.array(sorted(cols))
        local_keep = np.ones(rows.size, dtype=bool)
        matrix, values = design[np.ix_(rows, cols)], oc[rows]
        for _ in range(iterations):
            fit = matrix[local_keep]
            scale = np.linalg.norm(fit, axis=0)
            if fit.shape[0] <= fit.shape[1] or np.any(scale == 0):
                raise ValueError('insufficient empirical support after physical O-C QC')
            coefficients, _, rank, _ = lstsq(fit / scale, values[local_keep], cond=1e-12)
            if rank != fit.shape[1]:
                raise ValueError('rank deficient empirical model in physical O-C QC')
            detection = values[local_keep] - fit @ (coefficients / scale)
            tt = time[rows][local_keep]
            accepted = Outlier(config.get('OutlierTimes', 6),
                config.get('OutlierRMS'), config.get('OutlierObs')).remove(tt, detection)
            if config.get('EpochOutlierEnabled', True):
                epoch_keep, _ = EpochOutlier(config.get('EpochOutlierSignificance', .01),
                    config.get('EpochOutlierMaxFraction', .02),
                    config.get('EpochOutlierMinSamples', 20)).detect(tt, detection)
                accepted &= epoch_keep
            if accepted.all():
                break
            local_keep[np.flatnonzero(local_keep)[~accepted]] = False
        keep[rows] = local_keep
    if not keep.any():
        raise ValueError('physical O-C QC rejected every epoch')
    return keep
