# SPDX-License-Identifier: GPL-3.0-or-later
"""AOHIS/ESA-ESM stochastic prior training for PyHawk.

This module follows Kvas and Mayer-Gurr (2019): daily coefficients are formed
from aoErr + H + I when cryosphere input is enabled, long-period terms and
monthly means are removed, lag covariances are estimated month by month (in
spectral or partitioned EWH space), and multivariate Yule-Walker equations are
solved for a GROOPS-compatible AR model.
"""

from __future__ import annotations

import os
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np


PathLike = Union[str, os.PathLike]


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_epoch(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        value = np.datetime_as_string(value, unit="s")
    text = _decode(value).strip()
    iso_text = text.rstrip("Z")
    try:
        return datetime.fromisoformat(iso_text)
    except ValueError:
        pass
    for pattern in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d%H%M%S",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            pass
    match = re.search(r"((?:19|20)\d{2})[-_]?(\d{2})[-_]?(\d{2})", text)
    if match is None:
        raise ValueError("cannot parse epoch from %r" % text)
    year, month, day = (int(item) for item in match.groups())
    hour = minute = second = 0
    tail = text[match.end():]
    time_match = re.search(r"(?:T|[_ -])?(\d{2})[:_-]?(\d{2})(?:[:_-]?(\d{2}))?", tail)
    if time_match:
        hour, minute = int(time_match.group(1)), int(time_match.group(2))
        second = int(time_match.group(3) or 0)
    return datetime(year, month, day, hour, minute, second)


def _epoch_label(epoch: datetime) -> str:
    return epoch.strftime("%Y-%m-%dT%H:%M:%S")


def _triangular_size(degree_max: int) -> int:
    return (degree_max + 1) * (degree_max + 2) // 2


def _state_size(degree_max: int, degree_min: int) -> int:
    return (
        (degree_max + 1) * (degree_max + 2)
        - degree_min * (degree_min + 1)
        - (degree_max - degree_min + 1)
    )


def _to_state(
    values: np.ndarray,
    degree_max: int,
    degree_min: int,
    state_order: str,
) -> np.ndarray:
    """Convert full C/S arrays or already sorted rows to PyHawk state order."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    expected_state = _state_size(degree_max, degree_min)
    triangular = _triangular_size(degree_max)
    if array.ndim == 3 and array.shape[1] == 2:
        c, s = array[:, 0, :], array[:, 1, :]
    elif array.ndim == 2 and array.shape[1] == 2 * triangular:
        c, s = array[:, :triangular], array[:, triangular:]
    elif array.ndim == 2 and array.shape[1] == expected_state:
        return array.copy()
    else:
        raise ValueError(
            "AOHIS component must be (time,%d), (time,2,%d), or sorted state (time,%d)"
            % (2 * triangular, triangular, expected_state)
        )

    from src.Auxilary.SortCS import SortCS

    sorter = SortCS(degree_max, degree_min, method=state_order)
    return np.vstack([sorter.sort(c_row, s_row) for c_row, s_row in zip(c, s)])


def _find_key(container: Any, requested: str, aliases: Sequence[str]) -> str:
    candidates = [requested] + list(aliases)
    available = {str(key).lower(): str(key) for key in container.keys()}
    for candidate in candidates:
        if candidate and candidate in container:
            return candidate
        if candidate and candidate.lower() in available:
            return available[candidate.lower()]
    raise KeyError("none of the datasets %s exists" % candidates)


def _read_gfc_bytes(
    raw: bytes,
    name: str,
    degree_max: int,
) -> Tuple[datetime, np.ndarray]:
    """Read one ESA-ESM/ICGEM coefficient file from bytes."""

    triangular = _triangular_size(degree_max)
    c = np.zeros(triangular, dtype=np.float64)
    s = np.zeros(triangular, dtype=np.float64)
    epoch = None
    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].lower() in ("time_period_of_data", "time_coverage_start"):
            if epoch is None:
                try:
                    epoch = _parse_epoch(parts[1])
                except (IndexError, ValueError):
                    pass
        if parts[0].lower() not in ("gfc", "gfct") or len(parts) < 5:
            continue
        try:
            degree, order = int(parts[1]), int(parts[2])
            if degree > degree_max or order > degree:
                continue
            c[degree * (degree + 1) // 2 + order] = float(parts[3].replace("D", "E"))
            s[degree * (degree + 1) // 2 + order] = float(parts[4].replace("D", "E"))
        except (IndexError, ValueError):
            continue
    if epoch is None:
        epoch = _parse_epoch(name)
    return epoch, np.concatenate((c, s))


def _read_gfc_component(root: PathLike, degree_max: int) -> Dict[datetime, np.ndarray]:
    """Read a directory or tar archive containing one GFC field per epoch."""

    root_path = Path(root)

    def accepted(name: str) -> bool:
        return name.lower().endswith((".gfc", ".txt", ".dat", ".asc"))

    def archive(name: str) -> bool:
        return name.lower().endswith((".tar.gz", ".tgz", ".tar"))

    def payloads():
        if root_path.is_file() and archive(root_path.name):
            paths = [root_path]
        elif root_path.is_file():
            paths = []
            if accepted(root_path.name):
                with open(str(root_path), "rb") as file:
                    yield root_path.name, file.read()
            return
        else:
            paths = [
                path
                for path in sorted(root_path.rglob("*"))
                if path.is_file()
            ]

        for path in paths:
            if archive(path.name):
                with tarfile.open(str(path), "r:*") as archive_file:
                    for member in archive_file:
                        if member.isfile() and accepted(member.name):
                            handle = archive_file.extractfile(member)
                            if handle is not None:
                                yield member.name, handle.read()
            elif accepted(path.name):
                with open(str(path), "rb") as file:
                    yield str(path), file.read()

    result: Dict[datetime, np.ndarray] = {}
    for name, raw in payloads():
        epoch, values = _read_gfc_bytes(raw, name, degree_max)
        if epoch in result:
            raise ValueError("duplicate component epoch %s in %s" % (epoch, root_path))
        result[epoch] = values
    if not result:
        raise FileNotFoundError("no GFC files found in %s" % root_path)
    return result


def _daily_average(
    epochs: Sequence[datetime], values: np.ndarray
) -> Dict[datetime, np.ndarray]:
    grouped: Dict[datetime, List[int]] = {}
    for index, epoch in enumerate(epochs):
        day = datetime(epoch.year, epoch.month, epoch.day)
        grouped.setdefault(day, []).append(index)
    return {
        day: np.mean(values[indices], axis=0)
        for day, indices in sorted(grouped.items())
    }


@dataclass(frozen=True)
class EWHGrid:
    """EWH grid and its land/ocean partition used for AOHIS training."""

    latitude: np.ndarray
    longitude: np.ndarray
    land_mask: np.ndarray
    area_weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude, dtype=np.float64).reshape(-1)
        longitude = np.asarray(self.longitude, dtype=np.float64).reshape(-1)
        land_mask = np.asarray(self.land_mask, dtype=bool).reshape(-1)
        if latitude.size == 0 or latitude.size != longitude.size:
            raise ValueError("EWH latitude and longitude must have the same non-zero size")
        if land_mask.size != latitude.size:
            raise ValueError("EWH land_mask must match the grid size")
        if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
            raise ValueError("EWH grid coordinates must be finite")
        if np.any(latitude < -90.0) or np.any(latitude > 90.0):
            raise ValueError("EWH latitudes must be in [-90, 90] degrees")
        weights = self.area_weights
        if weights is None:
            weights = np.cos(np.deg2rad(latitude))
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.size != latitude.size or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("EWH area_weights must be finite and positive")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", np.mod(longitude, 360.0))
        object.__setattr__(self, "land_mask", land_mask)
        object.__setattr__(self, "area_weights", weights)

    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        latitude_dataset: str = "latitude",
        longitude_dataset: str = "longitude",
        land_mask_dataset: str = "land_mask",
        area_weights_dataset: str = "area_weights",
    ) -> "EWHGrid":
        """Load an NPZ or HDF5 grid file."""

        path = os.fspath(filename)
        extension = os.path.splitext(path)[1].lower()
        aliases = {
            "latitude": (latitude_dataset, "lat", "lats"),
            "longitude": (longitude_dataset, "lon", "lons"),
            "land_mask": (land_mask_dataset, "land", "mask"),
            "area_weights": (area_weights_dataset, "weights", "area"),
        }

        def find(container: Any, names: Sequence[str], required: bool = True):
            available = {str(key).lower(): key for key in container.keys()}
            for name in names:
                if name in container:
                    return container[name]
                if name.lower() in available:
                    return container[available[name.lower()]]
            if required:
                raise KeyError("none of the datasets %s exists in %s" % (names, path))
            return None

        if extension == ".npz":
            with np.load(path, allow_pickle=False) as data:
                latitude = find(data, aliases["latitude"])
                longitude = find(data, aliases["longitude"])
                land_mask = find(data, aliases["land_mask"])
                weights = find(data, aliases["area_weights"], required=False)
                return cls(latitude, longitude, land_mask, weights)
        with h5py.File(path, "r") as h5:
            latitude = find(h5, aliases["latitude"])[:]
            longitude = find(h5, aliases["longitude"])[:]
            land_mask = find(h5, aliases["land_mask"])[:]
            weights_dataset = find(h5, aliases["area_weights"], required=False)
            weights = weights_dataset[:] if weights_dataset is not None else None
        return cls(latitude, longitude, land_mask, weights)


def _state_column_description(
    degree_max: int,
    degree_min: int,
    state_order: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return degree, order and sine flags for every sorted state column."""

    from src.Auxilary.SortCS import SortCS

    triangular = _triangular_size(degree_max)
    sorter = SortCS(degree_max, degree_min, method=state_order)
    descriptions: List[Tuple[int, int, bool, int]] = []
    for degree in range(degree_min, degree_max + 1):
        for order in range(degree + 1):
            full_index = degree * (degree + 1) // 2 + order
            cosine = np.zeros(triangular)
            cosine[full_index] = 1.0
            sorted_column = sorter.sort(cosine, np.zeros(triangular))
            position = np.flatnonzero(sorted_column)
            if position.size != 1:
                raise ValueError("SortCS did not produce a one-to-one cosine state column")
            descriptions.append((int(position[0]), degree, order, False))
        for order in range(1, degree + 1):
            full_index = degree * (degree + 1) // 2 + order
            sine_column = np.zeros(triangular)
            sine_column[full_index] = 1.0
            sorted_column = sorter.sort(np.zeros(triangular), sine_column)
            position = np.flatnonzero(sorted_column)
            if position.size != 1:
                raise ValueError("SortCS did not produce a one-to-one sine state column")
            descriptions.append((int(position[0]), degree, order, True))
    descriptions.sort(key=lambda item: item[0])
    return (
        np.asarray([item[1] for item in descriptions], dtype=np.int64),
        np.asarray([item[2] for item in descriptions], dtype=np.int64),
        np.asarray([item[3] for item in descriptions], dtype=bool),
    )


def _ewh_synthesis_matrix(
    grid: EWHGrid,
    degree_max: int,
    degree_min: int,
    state_order: str,
    radius: float,
    earth_density: float,
    water_density: float,
    load_love_numbers: Optional[np.ndarray],
) -> np.ndarray:
    """Build EWH = S @ state using PyHawk's normalized harmonics."""

    if radius <= 0.0 or earth_density <= 0.0 or water_density <= 0.0:
        raise ValueError("radius and EWH densities must be positive")
    love = np.zeros(degree_max + 1, dtype=np.float64)
    if load_love_numbers is not None:
        supplied = np.asarray(load_love_numbers, dtype=np.float64).reshape(-1)
        if supplied.size < degree_max + 1:
            raise ValueError("load_love_numbers must contain degree 0 through degree_max")
        love[:] = supplied[: degree_max + 1]
    degrees, orders, sine = _state_column_description(
        degree_max, degree_min, state_order
    )
    scale = radius * earth_density / (3.0 * water_density) * (
        (2.0 * np.arange(degree_max + 1) + 1.0) / (1.0 + love)
    )
    longitudes = np.deg2rad(grid.longitude)
    unique_latitude, inverse = np.unique(grid.latitude, return_inverse=True)
    synthesis = np.empty((grid.latitude.size, degrees.size), dtype=np.float64)
    from src.Auxilary.GeoMathKit import GeoMathKit

    for latitude_index, latitude in enumerate(unique_latitude):
        pnm = GeoMathKit.Legendre(degree_max, np.deg2rad(latitude))
        rows = np.flatnonzero(inverse == latitude_index)
        values = pnm[degrees, orders] * scale[degrees]
        phase = orders[None, :] * longitudes[rows, None]
        synthesis[rows] = np.where(sine[None, :], np.sin(phase), np.cos(phase)) * values
    return synthesis


def _angular_distance_km(
    latitude_a: np.ndarray,
    longitude_a: np.ndarray,
    latitude_b: np.ndarray,
    longitude_b: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    """Return pairwise great-circle distances for two coordinate blocks."""

    lat_a = np.deg2rad(latitude_a)[:, None]
    lon_a = np.deg2rad(longitude_a)[:, None]
    lat_b = np.deg2rad(latitude_b)[None, :]
    lon_b = np.deg2rad(longitude_b)[None, :]
    cosine = (
        np.sin(lat_a) * np.sin(lat_b)
        + np.cos(lat_a) * np.cos(lat_b) * np.cos(lon_a - lon_b)
    )
    return radius_km * np.arccos(np.clip(cosine, -1.0, 1.0))


def _ewh_covariance_to_spectral(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    synthesis: np.ndarray,
    grid: EWHGrid,
    radius: float,
    distance_scale_km: Optional[float],
    distance_cutoff_km: Optional[float],
    block_size: int,
    analysis: Optional[np.ndarray] = None,
    denominator: Optional[float] = None,
    center_samples: bool = False,
    symmetrize: bool = True,
) -> np.ndarray:
    """Transform partitioned, distance-tapered EWH covariance to state space.

    The calculation is streamed by grid blocks. It does not create the global
    EWH covariance matrix, which would be prohibitively large for a one-degree
    global grid. The result is A (C_EWH * K) A.T, with A the area-weighted
    inverse of the synthesis matrix.
    """

    first = np.asarray(samples_a, dtype=np.float64)
    second = np.asarray(samples_b, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != grid.latitude.size:
        raise ValueError("EWH samples must be (time, grid_point)")
    if first.shape[0] != second.shape[0]:
        raise ValueError("lag covariance samples must have equal time length")
    if first.shape[0] < 2:
        raise ValueError("at least two samples are required for EWH covariance")
    if synthesis.shape[0] != grid.latitude.size:
        raise ValueError("EWH synthesis matrix does not match the grid")
    if block_size <= 0:
        raise ValueError("EWH block_size must be positive")
    if distance_scale_km is not None and distance_scale_km <= 0.0:
        raise ValueError("distance_scale_km must be positive")
    if distance_cutoff_km is not None and distance_cutoff_km <= 0.0:
        raise ValueError("distance_cutoff_km must be positive")

    if center_samples:
        centered_a = first - np.mean(first, axis=0, keepdims=True)
        centered_b = second - np.mean(second, axis=0, keepdims=True)
    else:
        centered_a = first
        centered_b = second
    covariance_denominator = (
        float(denominator) if denominator is not None else float(first.shape[0] - 1)
    )
    if covariance_denominator <= 0.0:
        raise ValueError("EWH covariance denominator must be positive")
    if analysis is None:
        weights = np.asarray(grid.area_weights, dtype=np.float64)
        weighted_synthesis = synthesis * weights[:, None]
        normal = synthesis.T @ weighted_synthesis
        inverse_normal = np.linalg.pinv(0.5 * (normal + normal.T), rcond=1.0e-12)
        analysis = inverse_normal @ weighted_synthesis.T
    state_size = synthesis.shape[1]
    result = np.zeros((state_size, state_size), dtype=np.float64)
    earth_radius_km = radius / 1000.0

    for region in (True, False):
        indices = np.flatnonzero(grid.land_mask == region)
        if indices.size == 0:
            continue
        for start in range(0, indices.size, block_size):
            row_indices = indices[start:start + block_size]
            rows_a = centered_a[:, row_indices]
            analysis_rows = analysis[:, row_indices]
            for column_start in range(0, indices.size, block_size):
                column_indices = indices[column_start:column_start + block_size]
                if (
                    distance_cutoff_km is not None
                    and distance_cutoff_km < 180.0 * np.pi * earth_radius_km
                ):
                    distance = _angular_distance_km(
                        grid.latitude[row_indices],
                        grid.longitude[row_indices],
                        grid.latitude[column_indices],
                        grid.longitude[column_indices],
                        earth_radius_km,
                    )
                    keep = distance <= distance_cutoff_km
                    if not np.any(keep):
                        continue
                else:
                    distance = None
                    keep = np.ones(
                        (row_indices.size, column_indices.size), dtype=bool
                    )
                covariance = (
                    rows_a.T @ centered_b[:, column_indices]
                    / covariance_denominator
                )
                if distance_scale_km is not None:
                    if distance is None:
                        distance = _angular_distance_km(
                            grid.latitude[row_indices],
                            grid.longitude[row_indices],
                            grid.latitude[column_indices],
                            grid.longitude[column_indices],
                            earth_radius_km,
                        )
                    covariance *= np.exp(-distance / distance_scale_km)
                covariance[~keep] = 0.0
                result += analysis_rows @ covariance @ analysis[:, column_indices].T
    return 0.5 * (result + result.T) if symmetrize else result


def _positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Add relative diagonal jitter only when roundoff breaks Cholesky."""

    value = 0.5 * (matrix + matrix.T)
    scale = max(abs(float(np.trace(value))) / max(value.shape[0], 1), np.finfo(float).tiny)
    for exponent in (0, -14, -12, -10, -8, -6):
        jitter = 0.0 if exponent == 0 else scale * (10.0 ** exponent)
        candidate = value + jitter * np.eye(value.shape[0])
        try:
            np.linalg.cholesky(candidate)
            return candidate
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("process covariance is not positive definite")


def _pseudo_observation(
    coefficients: Sequence[np.ndarray], process_covariance: np.ndarray
) -> np.ndarray:
    covariance = _positive_definite(process_covariance)
    lower = np.linalg.cholesky(covariance)
    whitening = np.linalg.solve(lower, np.eye(covariance.shape[0]))
    blocks = [-whitening @ coefficient for coefficient in reversed(coefficients)]
    blocks.append(whitening)
    return np.hstack(blocks)


def _yule_walker(
    lag_covariances: np.ndarray,
    order: int,
    ridge: float = 0.0,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Solve multivariate Yule-Walker equations using lag covariance blocks."""

    covariance_zero = 0.5 * (lag_covariances[0] + lag_covariances[0].T)
    dimension = covariance_zero.shape[0]
    if order == 0:
        return tuple(), _positive_definite(covariance_zero)

    predictor_covariance = np.zeros((order * dimension, order * dimension))
    for i in range(order):
        for j in range(order):
            lag = j - i
            if lag >= 0:
                block = lag_covariances[lag].T
            else:
                block = lag_covariances[-lag]
            predictor_covariance[
                i * dimension:(i + 1) * dimension,
                j * dimension:(j + 1) * dimension,
            ] = block
    predictor_covariance = 0.5 * (predictor_covariance + predictor_covariance.T)
    cross_covariance = np.hstack(
        [lag_covariances[index].T for index in range(1, order + 1)]
    )
    if ridge:
        scale = max(
            abs(float(np.trace(predictor_covariance))) / predictor_covariance.shape[0],
            np.finfo(float).tiny,
        )
        predictor_covariance = predictor_covariance + ridge * scale * np.eye(
            predictor_covariance.shape[0]
        )
    try:
        coefficient_block = np.linalg.solve(
            predictor_covariance, cross_covariance.T
        ).T
    except np.linalg.LinAlgError:
        coefficient_block = np.linalg.lstsq(
            predictor_covariance, cross_covariance.T, rcond=None
        )[0].T
    coefficients = tuple(
        coefficient_block[:, index * dimension:(index + 1) * dimension]
        for index in range(order)
    )
    process_covariance = covariance_zero - coefficient_block @ cross_covariance.T
    return coefficients, _positive_definite(process_covariance)


@dataclass
class AOHISTrainingResult:
    epochs: List[str]
    residuals: np.ndarray
    lag_covariances: np.ndarray
    coefficients: Tuple[np.ndarray, ...]
    process_covariance: np.ndarray
    pseudo_observation: np.ndarray
    models: Dict[int, np.ndarray]
    metadata: Dict[str, Any]

    def save(self, output_file: PathLike, include_residuals: bool = False) -> str:
        """Save the main GROOPS model plus diagnostics in HDF5 format."""

        output = os.fspath(output_file)
        parent = os.path.dirname(os.path.abspath(output))
        os.makedirs(parent, exist_ok=True)
        string_type = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output, "w") as h5:
            h5.create_dataset("model", data=self.pseudo_observation)
            dimension = self.process_covariance.shape[0]
            coefficients = (
                np.stack(self.coefficients)
                if self.coefficients
                else np.empty((0, dimension, dimension), dtype=np.float64)
            )
            h5.create_dataset("coefficients", data=coefficients)
            h5.create_dataset("process_covariance", data=self.process_covariance)
            h5.create_dataset("lag_covariances", data=self.lag_covariances)
            h5.create_dataset("epochs", data=np.asarray(self.epochs, dtype=string_type))
            if include_residuals:
                h5.create_dataset("residuals", data=self.residuals)
            for order, model in sorted(self.models.items()):
                h5.create_dataset("model_order_%d" % order, data=model)
            for key, value in self.metadata.items():
                if value is not None:
                    h5.attrs[key] = value
        return output


class AOHISTrainer:
    """Train the ESM-derived multivariate AR prior used by GROOPS."""

    def __init__(
        self,
        degree_max: int = 40,
        degree_min: int = 2,
        order: int = 3,
        state_order: str = "SortByOrder",
        monthly_mean_mode: str = "block",
        remove_trend: bool = True,
        remove_annual: bool = True,
        ridge: float = 0.0,
        warmup_models: bool = True,
        include_cryosphere: bool = False,
        covariance_domain: str = "spectral",
        ewh_grid: Optional[EWHGrid] = None,
        ewh_radius: float = 6.3781363e6,
        earth_density: float = 5517.0,
        water_density: float = 1000.0,
        load_love_numbers: Optional[np.ndarray] = None,
        distance_scale_km: Optional[float] = 1100.0,
        distance_cutoff_km: Optional[float] = 3300.0,
        ewh_block_size: int = 256,
    ):
        if degree_min < 0 or degree_min > degree_max:
            raise ValueError("degree_min must be in [0, degree_max]")
        if order < 0:
            raise ValueError("AR order must be non-negative")
        if monthly_mean_mode not in ("block", "climatology", "none"):
            raise ValueError("monthly_mean_mode must be block, climatology, or none")
        if covariance_domain not in ("spectral", "ewh"):
            raise ValueError("covariance_domain must be spectral or ewh")
        if covariance_domain == "ewh" and ewh_grid is None:
            raise ValueError("ewh_grid is required when covariance_domain=ewh")
        self.degree_max = int(degree_max)
        self.degree_min = int(degree_min)
        self.order = int(order)
        self.state_order = state_order
        self.monthly_mean_mode = monthly_mean_mode
        self.remove_trend = bool(remove_trend)
        self.remove_annual = bool(remove_annual)
        self.ridge = float(ridge)
        self.warmup_models = bool(warmup_models)
        self.include_cryosphere = bool(include_cryosphere)
        self.covariance_domain = covariance_domain
        self.ewh_grid = ewh_grid
        self.ewh_radius = float(ewh_radius)
        self.earth_density = float(earth_density)
        self.water_density = float(water_density)
        self.load_love_numbers = load_love_numbers
        self.distance_scale_km = distance_scale_km
        self.distance_cutoff_km = distance_cutoff_km
        self.ewh_block_size = int(ewh_block_size)
        self._ewh_synthesis: Optional[np.ndarray] = None
        self._ewh_analysis: Optional[np.ndarray] = None

    @property
    def state_size(self) -> int:
        return _state_size(self.degree_max, self.degree_min)

    def _remove_long_periods(
        self, dates: Sequence[datetime], values: np.ndarray
    ) -> np.ndarray:
        time = np.asarray(
            [(date - dates[0]).total_seconds() / 86400.0 / 365.25 for date in dates]
        )
        columns = [np.ones(len(dates))]
        if self.remove_trend:
            columns.append(time)
        if self.remove_annual:
            columns.extend((np.cos(2.0 * np.pi * time), np.sin(2.0 * np.pi * time)))
        design = np.column_stack(columns)
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residuals = values - design @ coefficients
        if self.monthly_mean_mode == "none":
            return residuals
        groups: Dict[Any, List[int]] = {}
        for index, date in enumerate(dates):
            key = (date.year, date.month) if self.monthly_mean_mode == "block" else date.month
            groups.setdefault(key, []).append(index)
        for indices in groups.values():
            residuals[indices] -= np.mean(residuals[indices], axis=0)
        return residuals

    def _lag_covariances(
        self, dates: Sequence[datetime], residuals: np.ndarray
    ) -> np.ndarray:
        groups: Dict[Tuple[int, int], List[int]] = {}
        for index, date in enumerate(dates):
            groups.setdefault((date.year, date.month), []).append(index)
        result = np.zeros((self.order + 1, self.state_size, self.state_size))
        if self.covariance_domain == "ewh":
            if self.ewh_grid is None:
                raise ValueError("EWH grid is required for EWH covariance training")
            if self._ewh_synthesis is None:
                self._ewh_synthesis = _ewh_synthesis_matrix(
                    self.ewh_grid,
                    self.degree_max,
                    self.degree_min,
                    self.state_order,
                    self.ewh_radius,
                    self.earth_density,
                    self.water_density,
                    self.load_love_numbers,
                )
            weights = self.ewh_grid.area_weights
            normal = self._ewh_synthesis.T @ (
                self._ewh_synthesis * weights[:, None]
            )
            inverse_normal = np.linalg.pinv(
                0.5 * (normal + normal.T), rcond=1.0e-12
            )
            self._ewh_analysis = inverse_normal @ (
                self._ewh_synthesis * weights[:, None]
            ).T
        for lag in range(self.order + 1):
            valid_months = 0
            for indices in groups.values():
                if len(indices) <= lag:
                    continue
                sample = residuals[indices]
                if self.covariance_domain == "ewh":
                    if lag == 0:
                        first, second = sample, sample
                    else:
                        first, second = sample[:-lag], sample[lag:]
                    ewh_first = first @ self._ewh_synthesis.T
                    ewh_second = second @ self._ewh_synthesis.T
                    result[lag] += _ewh_covariance_to_spectral(
                        ewh_first,
                        ewh_second,
                        self._ewh_synthesis,
                        self.ewh_grid,
                        self.ewh_radius,
                        self.distance_scale_km,
                        self.distance_cutoff_km,
                        self.ewh_block_size,
                        analysis=self._ewh_analysis,
                        denominator=len(indices) if lag == 0 else len(indices) - lag,
                        symmetrize=lag == 0,
                    )
                elif lag == 0:
                    result[lag] += sample.T @ sample / len(indices)
                else:
                    result[lag] += sample[:-lag].T @ sample[lag:] / (len(indices) - lag)
                valid_months += 1
            if valid_months == 0:
                raise ValueError("no month contains enough samples for lag %d" % lag)
            result[lag] /= valid_months
        result[0] = 0.5 * (result[0] + result[0].T)
        return result

    def fit(
        self,
        epochs: Sequence[Any],
        ao_error: np.ndarray,
        hydrology: Optional[np.ndarray] = None,
        cryosphere: Optional[np.ndarray] = None,
    ) -> AOHISTrainingResult:
        """Train from aligned six-hourly or daily aoErr, H and optional I arrays."""

        parsed_epochs = [_parse_epoch(value) for value in epochs]
        ao_state = _to_state(
            ao_error, self.degree_max, self.degree_min, self.state_order
        )
        if hydrology is None:
            hydro_state = np.zeros_like(ao_state)
        else:
            hydro_state = _to_state(
                hydrology, self.degree_max, self.degree_min, self.state_order
            )
        if self.include_cryosphere and cryosphere is None:
            raise ValueError(
                "cryosphere input I is required when include_cryosphere=True"
            )
        if cryosphere is None:
            ice_state = np.zeros_like(ao_state)
        else:
            ice_state = _to_state(
                cryosphere, self.degree_max, self.degree_min, self.state_order
            )
        if (
            len(parsed_epochs) != ao_state.shape[0]
            or ao_state.shape != hydro_state.shape
            or ao_state.shape != ice_state.shape
        ):
            raise ValueError(
                "epochs, ao_error, hydrology and cryosphere dimensions do not match"
            )

        order = np.asarray(
            sorted(range(len(parsed_epochs)), key=parsed_epochs.__getitem__),
            dtype=np.int64,
        )
        parsed_epochs = [parsed_epochs[index] for index in order]
        ao_state = ao_state[order]
        hydro_state = hydro_state[order]
        ice_state = ice_state[order]
        ao_daily = _daily_average(parsed_epochs, ao_state)
        hydro_daily = _daily_average(parsed_epochs, hydro_state)
        ice_daily = _daily_average(parsed_epochs, ice_state)
        common_days = sorted(set(ao_daily).intersection(hydro_daily))
        if self.include_cryosphere:
            common_days = sorted(set(common_days).intersection(ice_daily))
        if not common_days:
            raise ValueError("aoErr, H and I do not overlap in time")
        daily_values = np.vstack(
            [
                ao_daily[day] + hydro_daily[day] + ice_daily[day]
                for day in common_days
            ]
        )
        residuals = self._remove_long_periods(common_days, daily_values)
        lag_covariances = self._lag_covariances(common_days, residuals)

        coefficients_by_order: Dict[int, Tuple[np.ndarray, ...]] = {}
        covariance_by_order: Dict[int, np.ndarray] = {}
        model_orders = (
            range(self.order + 1) if self.warmup_models else (self.order,)
        )
        for current_order in model_orders:
            coefficients_by_order[current_order], covariance_by_order[current_order] = _yule_walker(
                lag_covariances, current_order, self.ridge
            )
        coefficients = coefficients_by_order[self.order]
        process_covariance = covariance_by_order[self.order]
        pseudo_observation = _pseudo_observation(coefficients, process_covariance)
        models = {
            current_order: (
                _pseudo_observation(coefficients_by_order[current_order], covariance_by_order[current_order])
                if current_order > 0
                else _pseudo_observation(tuple(), covariance_by_order[0])
            )
            for current_order in model_orders
        }
        metadata = {
            "degree_max": self.degree_max,
            "degree_min": self.degree_min,
            "state_dimension": self.state_size,
            "order": self.order,
            "component_definition": (
                "aoErr+H+I" if self.include_cryosphere else "aoErr+H"
            ),
            "covariance_domain": self.covariance_domain,
            "land_ocean_cross_covariance": "zero"
            if self.covariance_domain == "ewh"
            else None,
            "distance_taper": "exp(-distance/distance_scale_km)"
            if self.covariance_domain == "ewh" and self.distance_scale_km is not None
            else None,
            "distance_scale_km": self.distance_scale_km,
            "distance_cutoff_km": self.distance_cutoff_km,
            "sampling": "daily_mean_from_input_epochs",
            "monthly_mean_mode": self.monthly_mean_mode,
            "remove_trend": self.remove_trend,
            "remove_annual": self.remove_annual,
            "lag_covariance_estimator": "equal_weight_monthly_unbiased",
            "training_start": _epoch_label(common_days[0]),
            "training_end": _epoch_label(common_days[-1]),
        }
        return AOHISTrainingResult(
            epochs=[_epoch_label(day) for day in common_days],
            residuals=residuals,
            lag_covariances=lag_covariances,
            coefficients=coefficients,
            process_covariance=process_covariance,
            pseudo_observation=pseudo_observation,
            models=models,
            metadata=metadata,
        )

    def fit_from_file(
        self,
        input_file: PathLike,
        epoch_dataset: str = "epochs",
        ao_error_dataset: str = "aoErr",
        hydrology_dataset: str = "H",
        cryosphere_dataset: str = "I",
    ) -> AOHISTrainingResult:
        """Train from NPZ or HDF5 arrays named epochs, aoErr, H and I."""

        filename = os.fspath(input_file)
        extension = os.path.splitext(filename)[1].lower()
        if extension == ".npz":
            with np.load(filename, allow_pickle=False) as data:
                epoch_key = _find_key(data, epoch_dataset, ("epoch", "time", "dates"))
                ao_key = _find_key(data, ao_error_dataset, ("AOerr", "ao_error", "aod_error"))
                hydro_key = _find_key(data, hydrology_dataset, ("hydrology", "hydro"))
                ice_key = _find_key(
                    data, cryosphere_dataset, ("cryosphere", "ice", "I")
                ) if self.include_cryosphere else None
                return self.fit(
                    data[epoch_key],
                    data[ao_key],
                    data[hydro_key],
                    data[ice_key] if ice_key is not None else None,
                )
        with h5py.File(filename, "r") as h5:
            epoch_key = _find_key(h5, epoch_dataset, ("epoch", "time", "dates"))
            ao_key = _find_key(h5, ao_error_dataset, ("AOerr", "ao_error", "aod_error"))
            hydro_key = _find_key(h5, hydrology_dataset, ("hydrology", "hydro"))
            ice_key = _find_key(
                h5, cryosphere_dataset, ("cryosphere", "ice", "I")
            ) if self.include_cryosphere else None
            return self.fit(
                h5[epoch_key][:],
                h5[ao_key][:],
                h5[hydro_key][:],
                h5[ice_key][:] if ice_key is not None else None,
            )

    def fit_from_esm_directories(
        self,
        ao_error_directory: PathLike,
        hydrology_directory: PathLike,
        cryosphere_directory: Optional[PathLike] = None,
    ) -> AOHISTrainingResult:
        """Train from ESA ESM mtmdeal/AOerr, mtmshc/H and optional I."""

        ao_records = _read_gfc_component(ao_error_directory, self.degree_max)
        hydro_records = _read_gfc_component(hydrology_directory, self.degree_max)
        if self.include_cryosphere and not cryosphere_directory:
            raise ValueError(
                "cryosphere_directory is required when include_cryosphere=True"
            )
        ice_records = (
            _read_gfc_component(cryosphere_directory, self.degree_max)
            if cryosphere_directory
            else {}
        )
        common = sorted(set(ao_records).intersection(hydro_records))
        if self.include_cryosphere:
            common = sorted(set(common).intersection(ice_records))
        if not common:
            raise ValueError("AOerr, H and I directories have no common epochs")
        return self.fit(
            common,
            np.vstack([ao_records[epoch] for epoch in common]),
            np.vstack([hydro_records[epoch] for epoch in common]),
            np.vstack([ice_records[epoch] for epoch in common])
            if self.include_cryosphere
            else None,
        )


def train_aohis_from_config(options: Mapping[str, Any]) -> str:
    """Train and save a model using the flat PyHawk Kalman configuration."""

    covariance_domain = options.get("AOHISCovarianceDomain", "spectral")
    ewh_grid = None
    grid_file = options.get("AOHISEWHGridFile")
    if covariance_domain == "ewh":
        if not grid_file:
            raise ValueError(
                "AOHISEWHGridFile is required when AOHISCovarianceDomain=ewh"
            )
        ewh_grid = EWHGrid.from_file(
            grid_file,
            latitude_dataset=options.get("AOHISEWHLatitudeDataset", "latitude"),
            longitude_dataset=options.get("AOHISEWHLongitudeDataset", "longitude"),
            land_mask_dataset=options.get("AOHISLandMaskDataset", "land_mask"),
            area_weights_dataset=options.get("AOHISAreaWeightsDataset", "area_weights"),
        )
    love_numbers = None
    love_file = options.get("AOHISLoadLoveNumbersFile")
    if love_file:
        extension = os.path.splitext(os.fspath(love_file))[1].lower()
        if extension in (".h5", ".hdf5"):
            love_numbers = load_numeric_vector(
                love_file, options.get("AOHISLoadLoveNumbersDataset")
            )
        else:
            love_numbers = np.asarray(np.loadtxt(os.fspath(love_file)), dtype=np.float64)
    trainer = AOHISTrainer(
        degree_max=options.get("AOHISDegreeMax", 40),
        degree_min=options.get("AOHISDegreeMin", 2),
        order=options.get("AOHISOrder", 3),
        state_order=options.get("AOHISStateOrder", "SortByOrder"),
        monthly_mean_mode=options.get("AOHISMonthlyMeanMode", "block"),
        remove_trend=options.get("AOHISRemoveTrend", True),
        remove_annual=options.get("AOHISRemoveAnnual", True),
        ridge=options.get("AOHISRidge", 0.0),
        warmup_models=options.get("AOHISWarmupModels", True),
        include_cryosphere=options.get("AOHISIncludeCryosphere", False),
        covariance_domain=covariance_domain,
        ewh_grid=ewh_grid,
        ewh_radius=options.get("AOHISEWHRadius", 6.3781363e6),
        earth_density=options.get("AOHISEarthDensity", 5517.0),
        water_density=options.get("AOHISWaterDensity", 1000.0),
        load_love_numbers=love_numbers,
        distance_scale_km=options.get("AOHISDistanceScaleKm", 1100.0),
        distance_cutoff_km=options.get("AOHISDistanceCutoffKm", 3300.0),
        ewh_block_size=options.get("AOHISEWHBlockSize", 256),
    )
    ao_directory = options.get("AOHISAOErrorDirectory")
    hydro_directory = options.get("AOHISHydrologyDirectory")
    ice_directory = options.get("AOHISCryosphereDirectory")
    if ao_directory and hydro_directory:
        result = trainer.fit_from_esm_directories(
            ao_directory, hydro_directory, ice_directory
        )
    else:
        input_file = options.get("AOHISInputFile")
        if not input_file:
            raise ValueError(
                "set AOHISInputFile or both AOHISAOErrorDirectory and "
                "AOHISHydrologyDirectory"
            )
        result = trainer.fit_from_file(
            input_file,
            epoch_dataset=options.get("AOHISEpochDataset", "epochs"),
            ao_error_dataset=options.get("AOHISAOErrorDataset", "aoErr"),
            hydrology_dataset=options.get("AOHISHydrologyDataset", "H"),
            cryosphere_dataset=options.get("AOHISCryosphereDataset", "I"),
        )
    output_file = options.get(
        "AOHISModelOutputFile", "../result/Kalman/aohis_ar.hdf5"
    )
    return result.save(
        output_file,
        include_residuals=options.get("AOHISSaveResiduals", False),
    )


def load_numeric_vector(path: PathLike, dataset: Optional[str] = None) -> np.ndarray:
    """Load a one-dimensional vector from the same formats as model matrices."""

    filename = os.fspath(path)
    extension = os.path.splitext(filename)[1].lower()
    if extension == ".npy":
        return np.asarray(np.load(filename), dtype=np.float64).reshape(-1)
    if extension == ".npz":
        with np.load(filename, allow_pickle=False) as data:
            key = dataset or next(iter(data.files))
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    if extension in (".h5", ".hdf5"):
        if not dataset:
            raise ValueError("an HDF5 dataset name is required for %s" % filename)
        with h5py.File(filename, "r") as h5:
            return np.asarray(h5[dataset][:], dtype=np.float64).reshape(-1)
    return np.asarray(np.loadtxt(filename), dtype=np.float64).reshape(-1)


__all__ = [
    "AOHISTrainer",
    "AOHISTrainingResult",
    "EWHGrid",
    "load_numeric_vector",
    "train_aohis_from_config",
]
