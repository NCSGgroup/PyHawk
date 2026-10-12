# SPDX-License-Identifier: GPL-3.0-or-later
"""GROOPS-compatible information-form Kalman filtering for PyHawk.

This module adapts the numerical behavior of GROOPS ``KalmanFilter`` and
``AutoregressiveModel`` to NumPy.  It deliberately stays independent from
PyHawk's production driver: callers can pass normal equations as arrays or use
the HDF5 adapter for PyHawk ``NMatrix``/``LMatrix`` products.

Upstream sources (revision 7dbff92):
  - source/programs/kalmanFilter/kalmanFilter.cpp
  - source/misc/kalmanProcessing.cpp

Original GROOPS authors include Enrico Kurtenbach and Andreas Kvas.  GROOPS is
licensed under GPL-3.0-or-later; this adapted module retains that license.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]
PathLike = Union[str, os.PathLike]


def _as_square(matrix: ArrayLike, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("%s must be a square matrix" % name)
    if not np.all(np.isfinite(value)):
        raise ValueError("%s contains non-finite values" % name)
    return value


def _as_vector(vector: ArrayLike, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(value)):
        raise ValueError("%s contains non-finite values" % name)
    return value


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _solve(matrix: np.ndarray, right_hand_side: np.ndarray, allow_pseudoinverse: bool) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, right_hand_side)
    except np.linalg.LinAlgError:
        if not allow_pseudoinverse:
            raise
        return np.linalg.pinv(matrix) @ right_hand_side


@dataclass(frozen=True)
class NormalEquation:
    """One epoch of gravity-only normal equations ``N x = n``."""

    normal: np.ndarray
    right_hand_side: np.ndarray

    def __post_init__(self) -> None:
        normal = _as_square(self.normal, "normal")
        right_hand_side = _as_vector(self.right_hand_side, "right_hand_side")
        if normal.shape[0] != right_hand_side.size:
            raise ValueError("normal and right_hand_side dimensions do not match")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "right_hand_side", right_hand_side)


@dataclass(frozen=True)
class KalmanStep:
    """Predicted and updated state for one filter epoch."""

    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    updated_state: np.ndarray
    updated_covariance: np.ndarray
    measurement_used: bool
    transition: Optional[np.ndarray] = None
    process_covariance: Optional[np.ndarray] = None


class AutoregressiveModel:
    """Multivariate GROOPS AR(p) pseudo-observation model.

    GROOPS stores the model as ``[A_p ... A_1 A_0]`` such that
    ``A_0.T @ A_0`` is the process-noise precision matrix and
    ``Phi_k = -pinv(A_0) @ A_k``.
    """

    def __init__(self, pseudo_observation: ArrayLike):
        model = np.asarray(pseudo_observation, dtype=np.float64)
        if model.ndim != 2 or model.shape[0] == 0:
            raise ValueError("pseudo_observation must be a non-empty matrix")
        dimension = model.shape[0]
        if model.shape[1] % dimension != 0:
            raise ValueError("AR model column count must be a multiple of its row count")
        order = model.shape[1] // dimension - 1
        if order < 1:
            raise ValueError("Kalman filtering requires an AR model of order >= 1")
        if not np.all(np.isfinite(model)):
            raise ValueError("pseudo_observation contains non-finite values")
        self._model = model.copy()
        self._dimension = dimension
        self._order = order

    @classmethod
    def from_coefficients(cls, coefficients: Sequence[ArrayLike], process_covariance: ArrayLike):
        """Build GROOPS storage from ``[Phi_1, ..., Phi_p]`` and ``Q``."""

        if not coefficients:
            raise ValueError("at least one AR coefficient is required")
        covariance = _as_square(process_covariance, "process_covariance")
        dimension = covariance.shape[0]
        coefficient_matrices = [_as_square(item, "coefficient") for item in coefficients]
        if any(item.shape != (dimension, dimension) for item in coefficient_matrices):
            raise ValueError("all AR coefficients must match process_covariance")

        lower = np.linalg.cholesky(_symmetrize(covariance))
        whitening = np.linalg.solve(lower, np.eye(dimension))
        blocks = [-whitening @ item for item in reversed(coefficient_matrices)]
        blocks.append(whitening)
        return cls(np.hstack(blocks))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def order(self) -> int:
        return self._order

    @property
    def pseudo_observation(self) -> np.ndarray:
        return self._model.copy()

    def coefficients(self) -> Tuple[np.ndarray, ...]:
        """Return ``(Phi_1, ..., Phi_p)`` in prediction order."""

        dimension = self.dimension
        a0 = self._model[:, self.order * dimension :]
        inverse_weight = np.linalg.pinv(a0)
        oldest_to_newest = [
            -inverse_weight @ self._model[:, k * dimension : (k + 1) * dimension]
            for k in range(self.order)
        ]
        return tuple(reversed(oldest_to_newest))

    def white_noise_covariance(self) -> np.ndarray:
        """Return the AR prediction-error covariance ``Q``."""

        dimension = self.dimension
        a0 = self._model[:, self.order * dimension :]
        inverse_weight = np.linalg.pinv(a0)
        return _symmetrize(inverse_weight @ inverse_weight.T)

    def order_one_representation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert AR(p) to the companion-form AR(1) used by GROOPS."""

        coefficients = self.coefficients()
        process_covariance = self.white_noise_covariance()
        if self.order == 1:
            return coefficients[0].copy(), process_covariance

        size = self.order * self.dimension
        transition = np.zeros((size, size), dtype=np.float64)
        transition[: self.dimension, :] = np.hstack(coefficients)
        transition[self.dimension :, : -self.dimension] = np.eye(size - self.dimension)

        covariance = np.zeros((size, size), dtype=np.float64)
        covariance[: self.dimension, : self.dimension] = process_covariance
        return transition, covariance


def _warmup_model_matrices(
    models: Mapping[int, ArrayLike],
    max_order: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """Convert VAR(0)..VAR(p) pseudo-observations to common companion matrices."""

    if not models:
        raise ValueError("at least one autoregressive model is required")
    available = {int(order): np.asarray(value, dtype=np.float64) for order, value in models.items()}
    if 0 not in available:
        raise ValueError("warm-up model sequence must contain model_order_0")
    if max_order is None:
        max_order = max(available)
    max_order = int(max_order)
    if max_order < 1:
        raise ValueError("warm-up filtering requires a final AR order >= 1")
    model_zero = available[0]
    if model_zero.ndim != 2 or model_zero.shape[1] != model_zero.shape[0]:
        raise ValueError("VAR(0) model must be a square whitening matrix")
    dimension = model_zero.shape[0]
    if not np.all(np.isfinite(model_zero)):
        raise ValueError("warm-up model contains non-finite values")

    transitions: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    for requested_order in range(max_order + 1):
        order = min(requested_order, max_order)
        if order not in available:
            raise ValueError("missing model_order_%d in warm-up model sequence" % order)
        pseudo = available[order]
        if pseudo.ndim != 2 or pseudo.shape != (dimension, (order + 1) * dimension):
            raise ValueError(
                "model_order_%d must have shape (%d, %d)"
                % (order, dimension, (order + 1) * dimension)
            )
        a0 = pseudo[:, order * dimension:]
        inverse_weight = np.linalg.pinv(a0)
        coefficients = [
            -inverse_weight @ pseudo[:, (order - lag) * dimension:(order - lag + 1) * dimension]
            for lag in range(1, order + 1)
        ]
        process_covariance = _symmetrize(inverse_weight @ inverse_weight.T)
        transition = np.zeros((max_order * dimension, max_order * dimension))
        if order:
            transition[:dimension, :order * dimension] = np.hstack(coefficients)
        if max_order > 1:
            transition[dimension:, : (max_order - 1) * dimension] = np.eye(
                (max_order - 1) * dimension
            )
        covariance = np.zeros_like(transition)
        covariance[:dimension, :dimension] = process_covariance
        transitions.append(transition)
        covariances.append(covariance)
    return transitions, covariances, dimension


class InformationKalmanFilter:
    """Dense information-form filter matching GROOPS ``KalmanFilter``."""

    def __init__(
        self,
        transition: ArrayLike,
        process_covariance: ArrayLike,
        initial_covariance: ArrayLike,
        initial_state: Optional[ArrayLike] = None,
        observed_dimension: Optional[int] = None,
        allow_pseudoinverse: bool = False,
        transition_sequence: Optional[Sequence[ArrayLike]] = None,
        process_covariance_sequence: Optional[Sequence[ArrayLike]] = None,
    ):
        self.transition = _as_square(transition, "transition")
        self.process_covariance = _as_square(process_covariance, "process_covariance")
        self.updated_covariance = _as_square(initial_covariance, "initial_covariance").copy()
        size = self.transition.shape[0]
        if self.process_covariance.shape != (size, size):
            raise ValueError("process_covariance must match transition")
        if self.updated_covariance.shape != (size, size):
            raise ValueError("initial_covariance must match transition")

        if initial_state is None:
            self.updated_state = np.zeros(size, dtype=np.float64)
        else:
            self.updated_state = _as_vector(initial_state, "initial_state").copy()
            if self.updated_state.size != size:
                raise ValueError("initial_state must match transition")

        if observed_dimension is None:
            observed_dimension = size
        if observed_dimension <= 0 or observed_dimension > size:
            raise ValueError("observed_dimension must be in [1, state dimension]")
        self.observed_dimension = int(observed_dimension)
        self.allow_pseudoinverse = bool(allow_pseudoinverse)
        if (transition_sequence is None) != (process_covariance_sequence is None):
            raise ValueError(
                "transition_sequence and process_covariance_sequence must be supplied together"
            )
        if transition_sequence is None:
            self.transition_sequence = None
            self.process_covariance_sequence = None
        else:
            if len(transition_sequence) != len(process_covariance_sequence):
                raise ValueError("time-varying transition and covariance sequences must match")
            if not transition_sequence:
                raise ValueError("time-varying transition sequence must not be empty")
            self.transition_sequence = [
                _as_square(item, "transition_sequence item")
                for item in transition_sequence
            ]
            self.process_covariance_sequence = [
                _as_square(item, "process_covariance_sequence item")
                for item in process_covariance_sequence
            ]
            for transition_item, covariance_item in zip(
                self.transition_sequence, self.process_covariance_sequence
            ):
                if transition_item.shape != self.transition.shape:
                    raise ValueError("all transition_sequence matrices must have the same shape")
                if covariance_item.shape != self.process_covariance.shape:
                    raise ValueError(
                        "all process_covariance_sequence matrices must have the same shape"
                    )
        self._step_index = 0

    @classmethod
    def from_autoregressive_model(
        cls,
        model: AutoregressiveModel,
        initial_covariance: ArrayLike,
        initial_state: Optional[ArrayLike] = None,
        allow_pseudoinverse: bool = False,
        model_sequence: Optional[Mapping[int, ArrayLike]] = None,
    ):
        if model_sequence is not None:
            transitions, covariances, dimension = _warmup_model_matrices(model_sequence)
            state_dimension = transitions[-1].shape[0]
            supplied_covariance = np.asarray(initial_covariance, dtype=np.float64)
            if supplied_covariance.shape == (dimension, dimension):
                expanded_covariance = np.zeros((state_dimension, state_dimension))
                expanded_covariance[:dimension, :dimension] = supplied_covariance
                supplied_covariance = expanded_covariance
            supplied_state = None
            if initial_state is not None:
                raw_state = _as_vector(initial_state, "initial_state")
                if raw_state.size == dimension:
                    supplied_state = np.zeros(state_dimension)
                    supplied_state[:dimension] = raw_state
                else:
                    supplied_state = raw_state
            return cls(
                transition=transitions[-1],
                process_covariance=covariances[-1],
                initial_covariance=supplied_covariance,
                initial_state=supplied_state,
                observed_dimension=dimension,
                allow_pseudoinverse=allow_pseudoinverse,
                transition_sequence=transitions,
                process_covariance_sequence=covariances,
            )
        transition, process_covariance = model.order_one_representation()
        return cls(
            transition=transition,
            process_covariance=process_covariance,
            initial_covariance=initial_covariance,
            initial_state=initial_state,
            observed_dimension=model.dimension,
            allow_pseudoinverse=allow_pseudoinverse,
        )

    @property
    def state_dimension(self) -> int:
        return self.updated_state.size

    def _matrices_for_step(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.transition_sequence is None:
            return self.transition, self.process_covariance
        index = min(self._step_index, len(self.transition_sequence) - 1)
        return (
            self.transition_sequence[index],
            self.process_covariance_sequence[index],
        )

    def step(self, normal_equation: Optional[NormalEquation] = None) -> KalmanStep:
        """Predict one epoch and update it when a normal equation is present."""

        transition, process_covariance = self._matrices_for_step()
        predicted_state = transition @ self.updated_state
        predicted_covariance = _symmetrize(
            process_covariance
            + transition @ self.updated_covariance @ transition.T
        )

        if normal_equation is None:
            updated_state = predicted_state.copy()
            updated_covariance = predicted_covariance.copy()
            measurement_used = False
        else:
            normal = normal_equation.normal
            right_hand_side = normal_equation.right_hand_side
            observed = normal.shape[0]
            if observed != self.observed_dimension:
                raise ValueError(
                    "normal equation dimension %d does not match observed dimension %d"
                    % (observed, self.observed_dimension)
                )

            identity = np.eye(self.state_dimension)
            predicted_precision = _solve(
                predicted_covariance, identity, self.allow_pseudoinverse
            )
            updated_precision = _symmetrize(predicted_precision)
            updated_precision[:observed, :observed] += normal
            updated_covariance = _symmetrize(
                _solve(updated_precision, identity, self.allow_pseudoinverse)
            )

            innovation_normal = right_hand_side - normal @ predicted_state[:observed]
            updated_state = (
                predicted_state
                + updated_covariance[:, :observed] @ innovation_normal
            )
            measurement_used = True

        self.updated_state = updated_state
        self.updated_covariance = updated_covariance
        self._step_index += 1
        return KalmanStep(
            predicted_state=predicted_state,
            predicted_covariance=predicted_covariance,
            updated_state=updated_state,
            updated_covariance=updated_covariance,
            measurement_used=measurement_used,
            transition=transition.copy(),
            process_covariance=process_covariance.copy(),
        )

    def filter_iter(
        self, normal_equations: Iterable[Optional[NormalEquation]]
    ) -> Iterator[KalmanStep]:
        for normal_equation in normal_equations:
            yield self.step(normal_equation)

    def filter(
        self, normal_equations: Iterable[Optional[NormalEquation]]
    ) -> List[KalmanStep]:
        """Collect all steps; prefer ``filter_iter`` for degree-60 production runs."""

        return list(self.filter_iter(normal_equations))

    def smooth_steps(self, filtered_steps: Iterable[KalmanStep]) -> List[KalmanStep]:
        """Apply the Rauch-Tung-Striebel backward smoother to filter output.

        GROOPS publishes smoothed time-variable gravity solutions.  Keeping
        this operation separate from ``filter`` preserves the streaming API
        while allowing callers to choose a forward-only or smoothed solution.
        """

        steps = list(filtered_steps)
        if not steps:
            return []

        smoothed_state = steps[-1].updated_state.copy()
        smoothed_covariance = steps[-1].updated_covariance.copy()
        smoothed = [
            replace(
                steps[-1],
                updated_state=smoothed_state.copy(),
                updated_covariance=smoothed_covariance.copy(),
            )
        ]

        for index in range(len(steps) - 2, -1, -1):
            current = steps[index]
            next_step = steps[index + 1]
            identity = np.eye(self.state_dimension)
            predicted_inverse = _solve(
                next_step.predicted_covariance,
                identity,
                self.allow_pseudoinverse,
            )
            smoother_gain = (
                current.updated_covariance
                @ (
                    next_step.transition
                    if next_step.transition is not None
                    else self.transition
                ).T
                @ predicted_inverse
            )
            smoothed_state = current.updated_state + smoother_gain @ (
                smoothed_state - next_step.predicted_state
            )
            smoothed_covariance = _symmetrize(
                current.updated_covariance
                + smoother_gain
                @ (smoothed_covariance - next_step.predicted_covariance)
                @ smoother_gain.T
            )
            smoothed.append(
                replace(
                    current,
                    updated_state=smoothed_state.copy(),
                    updated_covariance=smoothed_covariance.copy(),
                )
            )

        smoothed.reverse()
        return smoothed

    def smooth(
        self, normal_equations: Iterable[Optional[NormalEquation]]
    ) -> List[KalmanStep]:
        """Filter all epochs and return their Rauch-Tung-Striebel solution."""

        return self.smooth_steps(self.filter(normal_equations))


def eliminate_local_parameters(
    normal_equation: NormalEquation,
    local_parameter_count: int,
    allow_pseudoinverse: bool = True,
) -> NormalEquation:
    """Eliminate a leading local-parameter block by Schur complement."""

    count = int(local_parameter_count)
    if count <= 0:
        return normal_equation
    normal = normal_equation.normal
    right_hand_side = normal_equation.right_hand_side
    if count >= normal.shape[0]:
        raise ValueError("local_parameter_count must be smaller than NEQ dimension")

    n_ll = normal[:count, :count]
    n_lg = normal[:count, count:]
    n_gl = normal[count:, :count]
    n_gg = normal[count:, count:]
    l_l = right_hand_side[:count]
    l_g = right_hand_side[count:]
    projection_normal = _solve(n_ll, n_lg, allow_pseudoinverse)
    projection_rhs = _solve(n_ll, l_l, allow_pseudoinverse)
    return NormalEquation(
        normal=_symmetrize(n_gg - n_gl @ projection_normal),
        right_hand_side=l_g - n_gl @ projection_rhs,
    )


def load_numeric_matrix(path: PathLike, dataset: Optional[str] = None) -> np.ndarray:
    """Load a dense matrix from NPY, NPZ, HDF5, or plain text."""

    filename = os.fspath(path)
    extension = os.path.splitext(filename)[1].lower()
    if extension == ".npy":
        return np.asarray(np.load(filename), dtype=np.float64)
    if extension == ".npz":
        with np.load(filename) as data:
            key = dataset or next(iter(data.files))
            return np.asarray(data[key], dtype=np.float64)
    if extension in (".h5", ".hdf5"):
        if not dataset:
            raise ValueError("an HDF5 dataset name is required for %s" % filename)
        with h5py.File(filename, "r") as h5:
            return np.asarray(h5[dataset][:], dtype=np.float64)
    return np.asarray(np.loadtxt(filename), dtype=np.float64)


def load_autoregressive_model_sequence(
    path: PathLike,
    dataset_prefix: str = "model_order_",
) -> Dict[int, np.ndarray]:
    """Load GROOPS-compatible VAR(0)..VAR(p) datasets from one HDF5 file."""

    filename = os.fspath(path)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in (".h5", ".hdf5"):
        raise ValueError("warm-up model sequences require an HDF5 model file")
    result: Dict[int, np.ndarray] = {}
    with h5py.File(filename, "r") as h5:
        for name in h5.keys():
            match = re.fullmatch(re.escape(dataset_prefix) + r"(\d+)", str(name))
            if match:
                result[int(match.group(1))] = np.asarray(h5[name][:], dtype=np.float64)
    if not result:
        raise KeyError("no %sN datasets found in %s" % (dataset_prefix, filename))
    return dict(sorted(result.items()))


def _normalise_epoch_label(value: Any, fallback_index: int) -> str:
    """Convert common date values and NEQ-name dates to ``YYYY-MM-DD``."""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.datetime64):
        value = np.datetime_as_string(value, unit="D")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    text = "" if value is None else str(value).strip()
    match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", text)
    if match is None:
        match = re.search(r"(\d{4})(\d{2})(\d{2})", text)
    if match is not None:
        year, month, day = (int(item) for item in match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return text or "epoch-%06d" % fallback_index


def _epoch_labels(
    files: Sequence[Optional[PathLike]], epochs: Optional[Iterable[Any]]
) -> List[str]:
    if epochs is not None:
        values = list(epochs)
        if len(values) != len(files):
            raise ValueError("epochs must have the same length as normal_equation_files")
        return [_normalise_epoch_label(value, index) for index, value in enumerate(values)]
    return [
        _normalise_epoch_label(os.fspath(path) if path is not None else None, index)
        for index, path in enumerate(files)
    ]


def _load_reference_field(
    reference_file: PathLike,
    degree_max: int,
    epoch: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read a static or time-dependent ICGEM field through PyHawk loaders."""

    from src.Interface.LoadSH import LoadGOCO06S, LoadGif48

    filename = os.fspath(reference_file)
    loader = None
    if epoch and re.match(r"^\d{4}-\d{2}-\d{2}$", epoch):
        try:
            loader = LoadGOCO06S().load(filename, date=epoch)
        except (IndexError, TypeError, ValueError):
            loader = None
    if loader is None:
        loader = LoadGif48().load(filename)

    c, s = loader.getCS(degree_max)
    sigma_c, sigma_s = loader.getSigmaCS(degree_max)
    expected = (degree_max + 1) * (degree_max + 2) // 2
    if len(c) != expected or len(s) != expected:
        raise ValueError(
            "reference field %s does not contain degree %d" % (filename, degree_max)
        )
    metadata = {
        "GM": loader.GM,
        "radius": loader.Radius,
        "norm": loader.norm or "fully_normalized",
        "tide_system": "zero_tide" if loader.zero_tide is not False else "tide_free",
    }
    return (
        np.asarray(c, dtype=np.float64),
        np.asarray(s, dtype=np.float64),
        np.asarray(sigma_c, dtype=np.float64),
        np.asarray(sigma_s, dtype=np.float64),
        metadata,
    )


def _load_optional_series_field(
    reference: PathLike,
    degree_max: int,
    epoch: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a static GFC or daily-average an ESA/GROOPS GFC time series."""

    path = Path(reference)
    is_archive = path.suffix.lower() in (".gz", ".tgz", ".tar")
    if not path.is_dir() and not is_archive:
        c, s, _, _, metadata = _load_reference_field(reference, degree_max, epoch)
        return c, s, metadata

    from src.Solver.AOHISTrainer import _read_gfc_component

    records = _read_gfc_component(reference, degree_max)
    daily: Dict[datetime, List[np.ndarray]] = {}
    for timestamp, values in records.items():
        day = datetime(timestamp.year, timestamp.month, timestamp.day)
        daily.setdefault(day, []).append(values)
    daily_values = {
        day: np.mean(np.vstack(values), axis=0) for day, values in daily.items()
    }
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(epoch))
    if date_match is None:
        raise ValueError("cannot select a time-series correction for epoch %s" % epoch)
    date = datetime(*(int(value) for value in date_match.groups()))
    if date not in daily_values:
        available = sorted(daily_values)
        date = min(available, key=lambda item: abs(item - date))
    values = daily_values[date]
    triangular = (degree_max + 1) * (degree_max + 2) // 2
    return values[:triangular], values[triangular:], {}


class ITSGGFCWriter:
    """Write daily Kalman states as ICGEM files with reference restoration.

    The state is interpreted as a Stokes correction in the configured
    ``SortCS`` order.  Degrees below ``degree_min`` are supplied only by the
    reference field, which mirrors the usual satellite-gravity product
    convention.  A reference file with ``gfct``, ``trnd``, ``acos`` and
    ``asin`` records is evaluated at every output epoch.
    """

    DEFAULT_GM = 3.986004415e14
    DEFAULT_RADIUS = 6.3781363e6

    def __init__(
        self,
        output_directory: PathLike,
        reference_gfc: Optional[PathLike],
        degree_max: int = 40,
        degree_min: int = 2,
        state_order: str = "SortByOrder",
        model_name: str = "PyHawk_Kalman",
        restore_reference: bool = True,
        gm: Optional[float] = None,
        radius: Optional[float] = None,
        norm: Optional[str] = None,
        tide_system: Optional[str] = None,
        aod1b_reference: Optional[PathLike] = None,
        gia_reference: Optional[PathLike] = None,
        degree1_reference: Optional[PathLike] = None,
        restore_aod1b: bool = False,
        remove_gia: bool = False,
        restore_degree1: bool = False,
    ):
        if degree_max < 0 or degree_min < 0 or degree_min > degree_max:
            raise ValueError("degree_min and degree_max must define a valid range")
        if restore_reference and not reference_gfc:
            raise ValueError(
                "reference_gfc is required when restore_reference=True; "
                "use restore_reference=False for a residual-only diagnostic"
            )
        if restore_aod1b and not aod1b_reference:
            raise ValueError(
                "aod1b_reference is required when restore_aod1b=True"
            )
        if remove_gia and not gia_reference:
            raise ValueError("gia_reference is required when remove_gia=True")
        if restore_degree1 and not degree1_reference:
            raise ValueError(
                "degree1_reference is required when restore_degree1=True"
            )
        if restore_degree1 and degree_max < 1:
            raise ValueError("degree_max must include degree 1 for degree-1 restoration")
        self.output_directory = Path(output_directory)
        self.reference_gfc = reference_gfc
        self.degree_max = int(degree_max)
        self.degree_min = int(degree_min)
        self.state_order = state_order
        self.model_name = model_name
        self.restore_reference = bool(restore_reference)
        self.gm = gm
        self.radius = radius
        self.norm = norm
        self.tide_system = tide_system
        self.aod1b_reference = aod1b_reference
        self.gia_reference = gia_reference
        self.degree1_reference = degree1_reference
        self.restore_aod1b = bool(restore_aod1b)
        self.remove_gia = bool(remove_gia)
        self.restore_degree1 = bool(restore_degree1)
        self._sort = None
        self._optional_series_cache: Dict[str, Dict[datetime, np.ndarray]] = {}

    @property
    def state_size(self) -> int:
        return (
            (self.degree_max + 1) * (self.degree_max + 2)
            - self.degree_min * (self.degree_min + 1)
            - (self.degree_max - self.degree_min + 1)
        )

    def _restore(
        self,
        state: np.ndarray,
        sigma_state: Optional[np.ndarray],
        epoch: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if state.size != self.state_size:
            raise ValueError(
                "state length %d does not match degree %d/%d SortCS layout (%d)"
                % (state.size, self.degree_min, self.degree_max, self.state_size)
            )
        if self._sort is None:
            from src.Auxilary.SortCS import SortCS

            self._sort = SortCS(
                degree_max=self.degree_max,
                degree_min=self.degree_min,
                method=self.state_order,
            )
        residual_c, residual_s = self._sort.invert(state)
        if sigma_state is None:
            residual_sigma_c = np.zeros_like(residual_c)
            residual_sigma_s = np.zeros_like(residual_s)
        else:
            residual_sigma_c, residual_sigma_s = self._sort.invert(sigma_state)

        if self.restore_reference:
            c, s, sigma_c, sigma_s, metadata = _load_reference_field(
                self.reference_gfc, self.degree_max, epoch
            )
        else:
            size = (self.degree_max + 1) * (self.degree_max + 2) // 2
            c = np.zeros(size, dtype=np.float64)
            s = np.zeros(size, dtype=np.float64)
            sigma_c = np.zeros(size, dtype=np.float64)
            sigma_s = np.zeros(size, dtype=np.float64)
            metadata = {}

        restored_components = ["reference_field"] if self.restore_reference else []
        if self.restore_aod1b and self.aod1b_reference:
            aod_c, aod_s, _ = _load_optional_series_field(
                self.aod1b_reference, self.degree_max, epoch
            )
            c += aod_c
            s += aod_s
            restored_components.append("AOD1B_daily_mean")
        if self.remove_gia and self.gia_reference:
            gia_c, gia_s, _ = _load_optional_series_field(
                self.gia_reference, self.degree_max, epoch
            )
            c -= gia_c
            s -= gia_s
            restored_components.append("minus_GIA")
        if self.restore_degree1 and self.degree1_reference:
            degree1_c, degree1_s, _ = _load_optional_series_field(
                self.degree1_reference, self.degree_max, epoch
            )
            # Degree-1 geocentre coefficients are C10, C11 and S11.  C00 is
            # deliberately left unchanged because it is a mass-conservation
            # convention, not a degree-1 geocentre term.
            c[1] += degree1_c[1]
            c[2] += degree1_c[2]
            s[2] += degree1_s[2]
            restored_components.append("degree1")

        metadata["GM"] = self.gm if self.gm is not None else metadata.get("GM")
        metadata["radius"] = (
            self.radius if self.radius is not None else metadata.get("radius")
        )
        metadata["norm"] = self.norm or metadata.get("norm") or "fully_normalized"
        metadata["tide_system"] = (
            self.tide_system
            or metadata.get("tide_system")
            or "zero_tide"
        )
        metadata["restored_components"] = ",".join(restored_components)
        c = c + residual_c
        s = s + residual_s
        return c, s, residual_sigma_c, residual_sigma_s, metadata

    def _write_one(
        self,
        epoch: str,
        c: np.ndarray,
        s: np.ndarray,
        sigma_c: np.ndarray,
        sigma_s: np.ndarray,
        metadata: Mapping[str, Any],
        state_dataset: str,
    ) -> Path:
        year = epoch[:4] if re.match(r"^\d{4}-\d{2}-\d{2}$", epoch) else ""
        directory = self.output_directory / year if year else self.output_directory
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / (
            "%s_n%d_%s.gfc" % (self.model_name, self.degree_max, epoch)
        )
        gm = metadata.get("GM") or self.DEFAULT_GM
        radius = metadata.get("radius") or self.DEFAULT_RADIUS
        norm = metadata.get("norm") or "fully_normalized"
        tide_system = metadata.get("tide_system") or "zero_tide"
        has_errors = bool(np.any(sigma_c) or np.any(sigma_s))
        errors = "formal" if has_errors else "no"
        fd, temporary = tempfile.mkstemp(
            prefix=filename.stem + ".", suffix=".tmp", dir=str(directory)
        )
        os.close(fd)
        try:
            with open(temporary, "w", encoding="utf-8", newline="\n") as file:
                file.write("product_type gravity_field\n")
                file.write("modelname %s\n" % filename.stem)
                file.write("earth_gravity_constant %.16e\n" % gm)
                file.write("radius %.16e\n" % radius)
                file.write("max_degree %d\n" % self.degree_max)
                file.write("norm %s\n" % norm)
                file.write("tide_system %s\n" % tide_system)
                file.write("errors %s\n" % errors)
                file.write("time_period_of_data %s - %s\n" % (epoch.replace("-", ""), epoch.replace("-", "")))
                file.write("generating_institute PyHawk\n")
                file.write("begin_of_head =============================================================\n")
                file.write("# epoch %s\n" % epoch)
                file.write("# source_dataset %s\n" % state_dataset)
                file.write("# reference_field %s\n" % (os.fspath(self.reference_gfc) if self.reference_gfc else "none"))
                file.write("# signal_definition reference_field_plus_daily_state\n")
                file.write(
                    "# restored_components %s\n"
                    % metadata.get("restored_components", "none")
                )
                file.write("# AOD1B is added as a daily mean when enabled\n")
                file.write("# GIA is subtracted when enabled; degree1 is added when enabled\n")
                file.write("# coefficients_below_degree_min supplied_by restored inputs\n")
                file.write("key L M C S sigmaC sigmaS\n")
                file.write("end_of_head ===============================================================\n")
                for order in range(self.degree_max + 1):
                    for degree in range(order, self.degree_max + 1):
                        index = degree * (degree + 1) // 2 + order
                        file.write(
                            "gfc %5d %5d % .16e % .16e % .16e % .16e\n"
                            % (
                                degree,
                                order,
                                c[index],
                                s[index],
                                sigma_c[index],
                                sigma_s[index],
                            )
                        )
            os.replace(temporary, str(filename))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return filename

    def write_from_hdf5(
        self,
        state_file: PathLike,
        state_dataset: str = "smoothed_state",
        epoch_dataset: str = "epoch",
        covariance_dataset: Optional[str] = None,
    ) -> List[str]:
        """Write one GFC file per epoch from the Kalman HDF5 product."""

        output_files = []
        with h5py.File(os.fspath(state_file), "r") as source:
            if state_dataset not in source:
                raise KeyError("missing state dataset %s" % state_dataset)
            states = source[state_dataset]
            if states.ndim != 2:
                raise ValueError("state dataset must be a two-dimensional time series")
            if epoch_dataset in source:
                epochs = [
                    _normalise_epoch_label(value, index)
                    for index, value in enumerate(source[epoch_dataset][:])
                ]
            elif "source_file" in source:
                epochs = [
                    _normalise_epoch_label(value, index)
                    for index, value in enumerate(source["source_file"][:])
                ]
            else:
                epochs = ["epoch-%06d" % index for index in range(states.shape[0])]

            if covariance_dataset is None:
                candidate = (
                    "smoothed_covariance"
                    if state_dataset == "smoothed_state"
                    else "updated_covariance"
                )
                covariance_dataset = candidate if candidate in source else None
            covariance = source[covariance_dataset] if covariance_dataset else None

            for index, epoch in enumerate(epochs):
                state = np.asarray(states[index], dtype=np.float64).reshape(-1)
                sigma_state = None
                if covariance is not None:
                    covariance_value = np.asarray(covariance[index], dtype=np.float64)
                    if covariance_value.shape != (state.size, state.size):
                        raise ValueError(
                            "covariance dataset must match the state dimension"
                        )
                    sigma_state = np.sqrt(
                        np.maximum(np.diag(covariance_value), 0.0)
                    )
                c, s, sigma_c, sigma_s, metadata = self._restore(
                    state, sigma_state, epoch
                )
                output_files.append(
                    str(
                        self._write_one(
                            epoch,
                            c,
                            s,
                            sigma_c,
                            sigma_s,
                            metadata,
                            state_dataset,
                        )
                    )
                )
        return output_files


def write_itsg_gfc_series(
    state_file: PathLike,
    output_directory: PathLike,
    reference_gfc: Optional[PathLike],
    degree_max: int = 40,
    degree_min: int = 2,
    state_order: str = "SortByOrder",
    state_dataset: str = "smoothed_state",
    covariance_dataset: Optional[str] = None,
    model_name: str = "PyHawk_Kalman",
    restore_reference: bool = True,
    aod1b_reference: Optional[PathLike] = None,
    gia_reference: Optional[PathLike] = None,
    degree1_reference: Optional[PathLike] = None,
    restore_aod1b: bool = False,
    remove_gia: bool = False,
    restore_degree1: bool = False,
) -> List[str]:
    """Convenience wrapper for writing an ITSG-like daily GFC series."""

    writer = ITSGGFCWriter(
        output_directory=output_directory,
        reference_gfc=reference_gfc,
        degree_max=degree_max,
        degree_min=degree_min,
        state_order=state_order,
        model_name=model_name,
        restore_reference=restore_reference,
        aod1b_reference=aod1b_reference,
        gia_reference=gia_reference,
        degree1_reference=degree1_reference,
        restore_aod1b=restore_aod1b,
        remove_gia=remove_gia,
        restore_degree1=restore_degree1,
    )
    return writer.write_from_hdf5(
        state_file=state_file,
        state_dataset=state_dataset,
        covariance_dataset=covariance_dataset,
    )


class PyHawkKalmanInterface:
    """Adapter between PyHawk HDF5 normal equations and the filter core."""

    def __init__(
        self,
        kalman_filter: InformationKalmanFilter,
        normal_dataset: str = "NMatrix",
        rhs_dataset: str = "LMatrix",
        local_parameter_count: int = 0,
    ):
        self.filter = kalman_filter
        self.normal_dataset = normal_dataset
        self.rhs_dataset = rhs_dataset
        self.local_parameter_count = int(local_parameter_count)

    @classmethod
    def from_groops_model(
        cls,
        pseudo_observation: ArrayLike,
        initial_covariance: ArrayLike,
        initial_state: Optional[ArrayLike] = None,
        normal_dataset: str = "NMatrix",
        rhs_dataset: str = "LMatrix",
        local_parameter_count: int = 0,
        allow_pseudoinverse: bool = False,
        model_sequence: Optional[Mapping[int, ArrayLike]] = None,
    ):
        model = AutoregressiveModel(pseudo_observation)
        kalman_filter = InformationKalmanFilter.from_autoregressive_model(
            model=model,
            initial_covariance=initial_covariance,
            initial_state=initial_state,
            allow_pseudoinverse=allow_pseudoinverse,
            model_sequence=model_sequence,
        )
        return cls(
            kalman_filter=kalman_filter,
            normal_dataset=normal_dataset,
            rhs_dataset=rhs_dataset,
            local_parameter_count=local_parameter_count,
        )

    def read_normal_equation(self, path: PathLike) -> NormalEquation:
        with h5py.File(os.fspath(path), "r") as h5:
            normal = h5[self.normal_dataset][:]
            right_hand_side = h5[self.rhs_dataset][:]
        normal_equation = NormalEquation(normal, right_hand_side)
        return eliminate_local_parameters(
            normal_equation,
            local_parameter_count=self.local_parameter_count,
            allow_pseudoinverse=True,
        )

    def run(
        self,
        normal_equation_files: Iterable[Optional[PathLike]],
        output_file: PathLike,
        save_covariance: bool = False,
        skip_missing: bool = True,
        epochs: Optional[Iterable[Any]] = None,
        smooth: bool = False,
    ) -> str:
        """Filter PyHawk daily NEQs and write a time series HDF5 product.

        ``smooth=True`` adds an RTS-smoothed state series while retaining the
        forward-filter result in ``updated_state``.  Epoch labels are either
        supplied explicitly or inferred from the NEQ file names.
        """

        files = list(normal_equation_files)
        epoch_labels = _epoch_labels(files, epochs)
        output_filename = os.fspath(output_file)
        output_directory = os.path.dirname(os.path.abspath(output_filename))
        os.makedirs(output_directory, exist_ok=True)
        epoch_count = len(files)
        observed = self.filter.observed_dimension
        string_type = h5py.string_dtype(encoding="utf-8")

        normal_equations = []
        for path in files:
            normal_equation = None
            if path is not None:
                source = os.fspath(path)
                if os.path.exists(source):
                    normal_equation = self.read_normal_equation(source)
                elif not skip_missing:
                    raise FileNotFoundError(source)
            normal_equations.append(normal_equation)

        filtered_steps = self.filter.filter(normal_equations)
        smoothed_steps = self.filter.smooth_steps(filtered_steps) if smooth else None

        with h5py.File(output_filename, "w") as output:
            updated = output.create_dataset(
                "updated_state", (epoch_count, observed), dtype=np.float64
            )
            predicted = output.create_dataset(
                "predicted_state", (epoch_count, observed), dtype=np.float64
            )
            used = output.create_dataset("measurement_used", (epoch_count,), dtype=np.bool_)
            sources = output.create_dataset("source_file", (epoch_count,), dtype=string_type)
            epoch_dataset = output.create_dataset("epoch", (epoch_count,), dtype=string_type)
            if smooth:
                smoothed = output.create_dataset(
                    "smoothed_state", (epoch_count, observed), dtype=np.float64
                )
            if save_covariance:
                updated_covariance = output.create_dataset(
                    "updated_covariance",
                    (epoch_count, observed, observed),
                    dtype=np.float64,
                )
                predicted_covariance = output.create_dataset(
                    "predicted_covariance",
                    (epoch_count, observed, observed),
                    dtype=np.float64,
                )
                if smooth:
                    smoothed_covariance = output.create_dataset(
                        "smoothed_covariance",
                        (epoch_count, observed, observed),
                        dtype=np.float64,
                    )

            output.attrs["state_dimension"] = self.filter.state_dimension
            output.attrs["observed_dimension"] = observed
            output.attrs["normal_dataset"] = self.normal_dataset
            output.attrs["rhs_dataset"] = self.rhs_dataset
            output.attrs["local_parameter_count"] = self.local_parameter_count
            output.attrs["algorithm"] = "GROOPS-compatible information Kalman filter"
            output.attrs["smoothed"] = bool(smooth)

            for index, path in enumerate(files):
                source = ""
                if path is not None:
                    source = os.fspath(path)
                step = filtered_steps[index]
                updated[index] = step.updated_state[:observed]
                predicted[index] = step.predicted_state[:observed]
                used[index] = step.measurement_used
                sources[index] = source
                epoch_dataset[index] = epoch_labels[index]
                if save_covariance:
                    updated_covariance[index] = step.updated_covariance[:observed, :observed]
                    predicted_covariance[index] = step.predicted_covariance[:observed, :observed]
                if smooth:
                    smooth_step = smoothed_steps[index]
                    smoothed[index] = smooth_step.updated_state[:observed]
                    if save_covariance:
                        smoothed_covariance[index] = smooth_step.updated_covariance[
                            :observed, :observed
                        ]

        return output_filename


def run_pyhawk_kalman(config: Union[PathLike, Mapping[str, Any]]) -> Optional[str]:
    """Reserved PyHawk entry point driven by ``KalmanFilterConfig.json``."""

    if isinstance(config, (str, os.PathLike)):
        with open(os.fspath(config), "r") as file:
            options: Dict[str, Any] = json.load(file)
    else:
        options = dict(config)
    if not options.get("Enabled", True):
        return None

    if options.get("TrainAOHIS", False):
        # Keep the trainer optional so the existing filter-only workflow stays
        # independent from the ESA ESM input dependencies.
        from src.Solver.AOHISTrainer import train_aohis_from_config

        trained_model_file = train_aohis_from_config(options)
        options["AutoregressiveModelFile"] = trained_model_file
        options["AutoregressiveModelDataset"] = "model"

    ar_model = load_numeric_matrix(
        options["AutoregressiveModelFile"],
        options.get("AutoregressiveModelDataset"),
    )
    initial_covariance = load_numeric_matrix(
        options["InitialCovarianceFile"],
        options.get("InitialCovarianceDataset"),
    )
    initial_state_file = options.get("InitialStateFile")
    initial_state = None
    if initial_state_file:
        initial_state = load_numeric_matrix(
            initial_state_file, options.get("InitialStateDataset")
        )

    model_sequence = None
    if options.get("AOHISUseWarmupModels", options.get("AOHISWarmupModels", False)):
        model_sequence_file = options.get("AOHISModelOutputFile") if options.get("TrainAOHIS") else options.get(
            "AutoregressiveModelFile"
        )
        if model_sequence_file:
            try:
                model_sequence = load_autoregressive_model_sequence(
                    model_sequence_file,
                    options.get("AOHISWarmupDatasetPrefix", "model_order_"),
                )
            except KeyError:
                if options.get("AOHISRequireWarmupModels", False):
                    raise

    interface = PyHawkKalmanInterface.from_groops_model(
        pseudo_observation=ar_model,
        initial_covariance=initial_covariance,
        initial_state=initial_state,
        normal_dataset=options.get("NormalDataset", "NMatrix"),
        rhs_dataset=options.get("RightHandSideDataset", "LMatrix"),
        local_parameter_count=options.get("LocalParameterCount", 0),
        allow_pseudoinverse=options.get("AllowPseudoInverse", False),
        model_sequence=model_sequence,
    )
    smooth = bool(options.get("Smooth", False))
    output_file = interface.run(
        normal_equation_files=options["NormalEquationFiles"],
        output_file=options["OutputFile"],
        save_covariance=options.get("SaveCovariance", False),
        skip_missing=options.get("SkipMissingNormalEquations", True),
        epochs=options.get("Epochs"),
        smooth=smooth,
    )
    gfc_directory = options.get("GFCOutputDirectory")
    if gfc_directory:
        state_dataset = options.get("GFCStateDataset") or (
            "smoothed_state" if smooth else "updated_state"
        )
        write_itsg_gfc_series(
            state_file=output_file,
            output_directory=gfc_directory,
            reference_gfc=options.get("GFCReferenceField") or None,
            degree_max=options.get("GFCDegreeMax", 40),
            degree_min=options.get("GFCDegreeMin", 2),
            state_order=options.get("GFCStateOrder", "SortByOrder"),
            state_dataset=state_dataset,
            covariance_dataset=options.get("GFCCovarianceDataset") or None,
            model_name=options.get("GFCModelName", "PyHawk_Kalman"),
            restore_reference=options.get("GFCRestoreReference", True),
            aod1b_reference=options.get("GFCAOD1BReference") or None,
            gia_reference=options.get("GFCGIAReference") or None,
            degree1_reference=options.get("GFCDegree1Reference") or None,
            restore_aod1b=options.get(
                "GFCRestoreAOD1B", bool(options.get("GFCAOD1BReference"))
            ),
            remove_gia=options.get(
                "GFCRemoveGIA", bool(options.get("GFCGIAReference"))
            ),
            restore_degree1=options.get(
                "GFCRestoreDegree1", bool(options.get("GFCDegree1Reference"))
            ),
        )
    return output_file


__all__ = [
    "AutoregressiveModel",
    "ITSGGFCWriter",
    "InformationKalmanFilter",
    "KalmanStep",
    "NormalEquation",
    "PyHawkKalmanInterface",
    "eliminate_local_parameters",
    "load_numeric_matrix",
    "load_autoregressive_model_sequence",
    "run_pyhawk_kalman",
    "write_itsg_gfc_series",
]
