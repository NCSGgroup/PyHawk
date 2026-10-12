"""Rank-checked, scaled least squares with genuinely shared local columns."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, solve_triangular, svd, eigh


@dataclass
class ObservationBlock:
    name: str
    global_dm: np.ndarray
    local_dm: np.ndarray
    residual: np.ndarray
    covariance: object = None  # covariance, NOT precision; None means identity
    weight: float = 1.0       # additional precision multiplier
    time: object = None

    def whiten(self):
        a = np.asarray(self.global_dm, dtype=float)
        b = np.asarray(self.local_dm, dtype=float)
        y = np.asarray(self.residual, dtype=float).reshape(-1)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != y.size or b.shape[0] != y.size:
            raise ValueError(self.name + ': inconsistent observation dimensions')
        if not y.size or not all(np.isfinite(x).all() for x in (a, b, y)):
            raise ValueError(self.name + ': empty or non-finite observations/design')
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('precision weight must be positive')
        matrix = np.column_stack((a, b, y))
        if self.covariance is not None:
            c = np.asarray(self.covariance, dtype=float)
            if not np.isfinite(c).all():
                raise ValueError('non-finite covariance')
            if c.ndim == 3:
                if c.shape[1:] != (3, 3) or c.shape[0] * 3 != y.size or not np.allclose(c, c.transpose(0,2,1), rtol=1e-10, atol=0):
                    raise ValueError('epoch covariance must contain symmetric 3x3 blocks')
                factors = np.linalg.cholesky(c)
                matrix = np.linalg.solve(factors, matrix.reshape(c.shape[0], 3, -1)).reshape(matrix.shape)
            elif c.ndim < 2:
                if c.ndim == 1 and c.shape != y.shape:
                    raise ValueError('diagonal covariance dimension mismatch')
                if np.any(c <= 0):
                    raise ValueError('covariance must be positive')
                matrix /= np.sqrt(c) if c.ndim == 0 else np.sqrt(c)[:, None]
            else:
                if c.shape != (y.size, y.size) or not np.allclose(c, c.T, rtol=1e-10, atol=0):
                    raise ValueError('covariance must be symmetric and match observations')
                matrix = solve_triangular(cholesky(c, lower=True), matrix, lower=True)
        matrix *= np.sqrt(self.weight)
        ng = a.shape[1]
        return matrix[:, :ng], matrix[:, ng:-1], matrix[:, -1]


@dataclass
class ReducedArc:
    normal: np.ndarray
    rhs: np.ndarray
    local_offset: np.ndarray
    local_global: np.ndarray
    local_rank: int
    observation_count: int

    def recover(self, global_delta):
        return self.local_offset - self.local_global @ global_delta


def reduce_arc(blocks, rcond=1e-12):
    """Stack BEFORE eliminating. Thin SVD avoids a full observation-size Q."""
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError('rcond must be in (0, 1)')
    if not blocks or len({b.name for b in blocks}) != len(blocks):
        raise ValueError('blocks must be nonempty with unique names')
    white = [b.whiten() for b in blocks]
    a, b, y = (np.concatenate([row[i] for row in white], axis=0) for i in range(3))
    nl = b.shape[1]
    if nl:
        scale = np.linalg.norm(b, axis=0)
        if np.any(scale == 0):
            raise ValueError('unobserved local column; remove it or add a physical constraint')
        u, singular, vt = svd(b / scale, full_matrices=False)
        rank = int(np.count_nonzero(singular > rcond * singular[0]))
        if rank != nl:
            raise ValueError('local design rank deficient: %d/%d; no unique physical feedback' % (rank, nl))
        ua, uy = u.T @ a, u.T @ y
        local_global = (vt.T @ (ua / singular[:, None])) / scale[:, None]
        local_offset = (vt.T @ (uy / singular)) / scale
        a = a - u @ ua
        y = y - u @ uy
    else:
        rank = 0
        local_global = np.empty((0, a.shape[1]))
        local_offset = np.empty(0)
    n = a.T @ a
    return ReducedArc((n + n.T) / 2, a.T @ y, local_offset,
                      local_global, rank, y.size)


def solve_global(normal, rhs, rcond=1e-12, allow_rank_deficient=False):
    """Equilibrated symmetric solve; report rank rather than hiding null modes."""
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError('rcond must be in (0, 1)')
    n, rhs = np.asarray(normal, float), np.asarray(rhs, float).reshape(-1)
    if not rhs.size:
        raise ValueError('no global parameters')
    if n.shape != (rhs.size, rhs.size) or not np.isfinite(n).all() or not np.isfinite(rhs).all():
        raise ValueError('invalid global normal equation')
    diagonal = np.diag(n)
    if np.any(diagonal <= 0):
        raise ValueError('unobserved global parameter')
    scale = np.sqrt(diagonal)
    ns = n / scale[:, None] / scale[None, :]
    values, vectors = eigh((ns + ns.T) / 2)
    threshold = rcond * values[-1]
    if values[0] < -max(threshold, 1e-12 * values[-1]):
        raise ValueError('global normal equation is not positive semidefinite')
    keep = values > threshold
    rank = int(keep.sum())
    if rank != rhs.size and not allow_rank_deficient:
        raise ValueError('global rank deficient: %d/%d; use more arcs or explicit constraints' % (rank, rhs.size))
    v = vectors[:, keep]
    delta = (v @ ((v.T @ (rhs / scale)) / values[keep])) / scale
    return delta, {'rank': rank, 'parameters': rhs.size,
                   'scaled_condition': float(values[-1] / values[keep][0]),
                   'rcond': rcond}


def postfit(block, global_delta, local_delta):
    return (np.asarray(block.residual).reshape(-1) - block.global_dm @ global_delta
            - block.local_dm @ local_delta)
