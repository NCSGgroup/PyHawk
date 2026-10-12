"""Core GLS utilities extracted from the PyHawk gravity recovery workflow.

This module is intentionally independent from ``CalibrateOrbit`` and ``NEQ``.
It collects the numerical core used by the manuscript-oriented GLS method:

1. residual-driven stochastic modeling,
2. positive spectral/covariance regularization, and
3. covariance-weighted least squares with local-parameter elimination.

The functions are not wired into the production driver yet; they are a clean
reference layer for refactoring the existing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class StochasticModelConfig:
    """Configuration for residual-to-covariance estimation."""

    dt: float
    max_lag: int
    eps_rel: float = 1e-4
    eps_abs: float = 1e-18
    remove_linear_trend: bool = True


@dataclass(frozen=True)
class NormalEquation:
    """Global-only normal equation block."""

    normal: np.ndarray
    right_hand_side: np.ndarray


def detrend_and_demean(residual: np.ndarray, remove_linear_trend: bool = True) -> np.ndarray:
    """Remove deterministic low-frequency content before stochastic modeling."""

    residual = np.asarray(residual, dtype=float).reshape(-1)
    if residual.size == 0:
        return residual
    if remove_linear_trend and residual.size > 1:
        t = np.arange(residual.size, dtype=float)
        design = np.column_stack((t, np.ones_like(t)))
        trend = design @ np.linalg.lstsq(design, residual, rcond=None)[0]
        residual = residual - trend
    return residual - np.mean(residual)


def uniform_grid_with_mask(time: np.ndarray, residual: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Map irregular or gapped residual samples onto an indexed uniform grid."""

    time = np.asarray(time, dtype=float).reshape(-1)
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if time.size != residual.size:
        raise ValueError("time and residual must have the same length")
    if time.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    t0 = np.min(time)
    index = np.round((time - t0) / dt).astype(np.int64)
    n = int(index.max()) + 1

    series = np.zeros(n, dtype=float)
    mask = np.zeros(n, dtype=float)
    series[index] = residual
    mask[index] = 1.0
    return series, mask


def masked_autocovariance(series: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
    """Estimate autocovariance with gap-aware pair-count normalization."""

    series = np.asarray(series, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=float).reshape(-1)
    if series.size != mask.size:
        raise ValueError("series and mask must have the same length")
    if series.size == 0:
        return np.zeros(max_lag + 1, dtype=float)

    nfft = 1 << (2 * series.size - 1).bit_length()
    x_fft = np.fft.fft(series, nfft)
    m_fft = np.fft.fft(mask, nfft)

    numerator = np.fft.ifft(x_fft * np.conj(x_fft)).real
    denominator = np.fft.ifft(m_fft * np.conj(m_fft)).real

    acov = np.zeros(max_lag + 1, dtype=float)
    valid = denominator[: max_lag + 1] > 0
    acov[valid] = numerator[: max_lag + 1][valid] / denominator[: max_lag + 1][valid]
    return acov


def positive_autocovariance(acov: np.ndarray, eps_rel: float = 1e-4, eps_abs: float = 1e-18) -> np.ndarray:
    """Clip the DCT-I spectrum and transform it back to a positive covariance model."""

    acov = np.asarray(acov, dtype=float).reshape(-1)
    if acov.size == 0:
        return acov

    eps = max(abs(acov[0]) * eps_rel, eps_abs)
    spectrum = dct1(acov)
    spectrum = np.maximum(spectrum, eps)
    return idct1(spectrum)


def dct1(values: np.ndarray) -> np.ndarray:
    """Compute an unnormalized DCT-I using NumPy FFT."""

    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size <= 1:
        return values.copy()
    mirrored = np.concatenate((values, values[-2:0:-1]))
    return np.fft.fft(mirrored).real[: values.size]


def idct1(values: np.ndarray) -> np.ndarray:
    """Compute the inverse of the unnormalized DCT-I used by :func:`dct1`."""

    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size <= 1:
        return values.copy()
    return dct1(values) / (2 * (values.size - 1))


def toeplitz_covariance(acov: np.ndarray, size: int) -> np.ndarray:
    """Build a Toeplitz covariance matrix from an autocovariance sequence."""

    acov = np.asarray(acov, dtype=float).reshape(-1)
    if acov.size < size:
        acov = np.pad(acov, (0, size - acov.size), mode="constant")
    index = np.arange(size)
    return acov[np.abs(index[:, None] - index[None, :])]


def covariance_from_residuals(
    time: np.ndarray,
    residual: np.ndarray,
    config: StochasticModelConfig,
) -> np.ndarray:
    """Estimate a full temporal covariance matrix from scalar residuals."""

    residual = detrend_and_demean(residual, config.remove_linear_trend)
    series, mask = uniform_grid_with_mask(time, residual, config.dt)
    acov = masked_autocovariance(series, mask, config.max_lag)
    acov = positive_autocovariance(acov, config.eps_rel, config.eps_abs)
    return toeplitz_covariance(acov, config.max_lag + 1)


def orbit_position_covariance(
    time: np.ndarray,
    residual_xyz: np.ndarray,
    config: StochasticModelConfig,
) -> np.ndarray:
    """Estimate interleaved 3D orbit-position covariance from x/y/z residuals."""

    residual_xyz = np.asarray(residual_xyz, dtype=float).reshape((-1, 3))
    component_cov = [
        covariance_from_residuals(time, residual_xyz[:, axis], config)
        for axis in range(3)
    ]

    n = component_cov[0].shape[0]
    covariance = np.zeros((3 * n, 3 * n), dtype=float)
    for axis, sigma_axis in enumerate(component_cov):
        covariance[axis::3, axis::3] = sigma_axis
    return covariance


def weighted_keep_global(
    dm_global: np.ndarray,
    dm_local: np.ndarray,
    obs: np.ndarray,
    covariance: np.ndarray | None = None,
) -> NormalEquation:
    """Build a global-only GLS normal equation after eliminating local parameters.

    The observation equation is ``obs = dm_global @ x_global + dm_local @ x_local``.
    If ``covariance`` is supplied, Cholesky whitening is applied, which is equivalent
    to weighted least squares with ``P = covariance^{-1}``.
    """

    dm_global = np.asarray(dm_global, dtype=float)
    dm_local = np.asarray(dm_local, dtype=float)
    obs = np.asarray(obs, dtype=float).reshape((-1, 1))

    if dm_global.shape[0] != obs.shape[0] or dm_local.shape[0] != obs.shape[0]:
        raise ValueError("design matrices and observation vector must have equal row counts")

    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        chol = np.linalg.cholesky(covariance)
        dm_global = np.linalg.solve(chol, dm_global)
        dm_local = np.linalg.solve(chol, dm_local)
        obs = np.linalg.solve(chol, obs)

    if dm_local.shape[1] == 0:
        projected_global = dm_global
        projected_obs = obs
    else:
        q_matrix, _ = np.linalg.qr(dm_local, mode="complete")
        null_projector = q_matrix[:, dm_local.shape[1] :].T
        projected_global = null_projector @ dm_global
        projected_obs = null_projector @ obs

    normal = projected_global.T @ projected_global
    right_hand_side = projected_global.T @ projected_obs
    return NormalEquation(normal=normal, right_hand_side=right_hand_side)


def accumulate_normal_equations(
    equations: Iterable[NormalEquation],
    weights: Iterable[float] | None = None,
) -> NormalEquation:
    """Accumulate monthly normal equations from arc-wise GLS blocks."""

    equations = list(equations)
    if not equations:
        raise ValueError("at least one normal equation is required")

    if weights is None:
        weights = [1.0] * len(equations)
    else:
        weights = list(weights)

    if len(weights) != len(equations):
        raise ValueError("weights and equations must have the same length")

    normal = np.zeros_like(equations[0].normal, dtype=float)
    right_hand_side = np.zeros_like(equations[0].right_hand_side, dtype=float)

    for weight, equation in zip(weights, equations):
        normal += weight * equation.normal
        right_hand_side += weight * equation.right_hand_side

    return NormalEquation(normal=normal, right_hand_side=right_hand_side)


def solve_global_parameters(equation: NormalEquation) -> np.ndarray:
    """Solve the accumulated monthly normal equation."""

    return np.linalg.lstsq(
        equation.normal,
        equation.right_hand_side[:, 0],
        rcond=None,
    )[0]
