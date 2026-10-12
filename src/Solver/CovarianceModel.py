import os

import h5py
import numpy as np


REPRESENTATION = 'stationary_autocovariance_v1'

def normalize_orbit_covariance_mode(mode):
    aliases = {
        'fixeddiagonal': 'FixedDiagonal',
        'diagonal': 'FixedDiagonal',
        'estimatedfull': 'EstimatedFull',
        'full': 'EstimatedFull',
        'itsgepoch': 'ITSGEpoch',
        'itsg': 'ITSGEpoch',
    }
    key = str(mode).replace('_', '').replace('-', '').lower()
    if key not in aliases:
        raise ValueError(
            'Unsupported OrbitCovarianceMode {!r}; expected FixedDiagonal, '
            'EstimatedFull, or ITSGEpoch'.format(mode)
        )
    return aliases[key]


def normalize_covariance_epoch_mapping(mode):
    aliases = {
        'legacyindex': 'LegacyIndex',
        'legacy': 'LegacyIndex',
        'physicaltime': 'PhysicalTime',
        'physical': 'PhysicalTime'
    }
    key = str(mode).replace('_', '').replace('-', '').lower()
    if key not in aliases:
        raise ValueError(
            'Unsupported CovarianceEpochMapping {!r}; expected '
            'LegacyIndex or PhysicalTime'.format(mode)
        )
    return aliases[key]


def covariance_matrix_epoch_index(mode, legacy_epoch_index):
    """Select legacy matrix indices or physical-time mapping explicitly."""

    if normalize_covariance_epoch_mapping(mode) == 'LegacyIndex':
        return legacy_epoch_index
    return None


def orbit_covariance_filename(post_orbit, date_span, satellite):
    directory = os.path.join(
        post_orbit, '{}-{}'.format(date_span[0], date_span[1])
    )
    return os.path.join(
        directory, '_covariance_Orbit_{}.hdf5'.format(satellite)
    )


def range_rate_covariance_filename(post_kbrr, date_span):
    directory = os.path.join(
        post_kbrr, '{}-{}'.format(date_span[0], date_span[1])
    )
    return os.path.join(directory, '_covariance_RangeRate.hdf5')


def save_stationary_covariance(
        filename, autocovariance=None, sample_interval=None,
        diagonal_value=None, time_origin=0.0,
        allow_nearest_epoch=False):
    """Save one shared, compact monthly covariance model."""

    has_autocovariance = autocovariance is not None
    has_diagonal = diagonal_value is not None
    if has_autocovariance == has_diagonal:
        raise ValueError(
            'provide exactly one of autocovariance or diagonal_value'
        )

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with h5py.File(filename, 'w') as h5:
        h5.attrs['representation'] = REPRESENTATION
        if has_diagonal:
            diagonal_value = float(diagonal_value)
            if not np.isfinite(diagonal_value) or diagonal_value <= 0.0:
                raise ValueError(
                    'covariance diagonal must be finite and positive'
                )
            h5.attrs['mode'] = 'diagonal'
            h5.attrs['diagonal_value'] = diagonal_value
            return

        autocovariance = np.asarray(autocovariance, dtype=float)
        if autocovariance.ndim == 1:
            autocovariance = autocovariance.reshape(1, -1)
        if autocovariance.ndim != 2 or autocovariance.shape[1] == 0:
            raise ValueError(
                'autocovariance must have shape (components, lag_count)'
            )
        if not np.isfinite(autocovariance).all():
            raise ValueError('autocovariance contains NaN or infinity')

        sample_interval = float(sample_interval)
        if not np.isfinite(sample_interval) or sample_interval <= 0.0:
            raise ValueError(
                'sample_interval must be finite and positive'
            )
        time_origin = float(time_origin)
        if not np.isfinite(time_origin):
            raise ValueError('time_origin must be finite')

        h5.attrs['mode'] = 'stationary'
        h5.attrs['sample_interval'] = sample_interval
        h5.attrs['time_origin'] = time_origin
        h5.attrs['epoch_mapping'] = (
            'nearest' if allow_nearest_epoch else 'exact'
        )
        h5.attrs['components'] = autocovariance.shape[0]
        h5.create_dataset(
            'autocovariance', data=autocovariance,
            compression='gzip', shuffle=True
        )


def _read_representation(h5, filename):
    representation = h5.attrs.get('representation', '')
    if isinstance(representation, bytes):
        representation = representation.decode('utf-8')
    if representation != REPRESENTATION:
        raise ValueError(
            'unsupported covariance representation in {}: {!r}'.format(
                filename, representation
            )
        )


def load_covariance(
        filename, target_time, components, matrix_epoch_index=None):
    """Materialize the covariance at the requested epochs.

    By default, the lag of each matrix element is computed from the physical
    epoch difference, so missing observations do not compress the covariance
    lag. An exact stationary model uses the first requested epoch as its
    index origin: a common time shift cannot change pairwise lags. The stored
    time_origin is retained for the legacy nearest-grid mapping only.
    ``matrix_epoch_index`` selects the stored Toeplitz model directly in
    index space.  The latter reproduces the legacy orbit-GLS operation
    ``Sigma[np.ix_(idx, idx)]`` without storing the dense ``Sigma`` matrix.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            'GLS covariance model does not exist: {}'.format(filename)
        )

    target_time = np.asarray(target_time, dtype=float).reshape(-1)
    if not np.isfinite(target_time).all():
        raise ValueError('target covariance epochs contain NaN or infinity')
    if np.unique(target_time).size != target_time.size:
        raise ValueError('target covariance epochs must be unique')
    if components not in (1, 3):
        raise ValueError('components must be either 1 or 3')

    if matrix_epoch_index is not None:
        matrix_epoch_index = np.asarray(
            matrix_epoch_index, dtype=float
        ).reshape(-1)
        if matrix_epoch_index.size != target_time.size:
            raise ValueError(
                'matrix_epoch_index and target_time must have equal length'
            )
        if not np.isfinite(matrix_epoch_index).all():
            raise ValueError(
                'matrix covariance indices contain NaN or infinity'
            )
        rounded_index = np.rint(matrix_epoch_index)
        if np.any(np.abs(matrix_epoch_index - rounded_index) > 1.0e-9):
            raise ValueError('matrix covariance indices must be integers')
        epoch_index = rounded_index.astype(np.int64)
        if np.any(epoch_index < 0):
            raise ValueError(
                'matrix covariance indices must be non-negative'
            )
        if np.unique(epoch_index).size != epoch_index.size:
            raise ValueError('matrix covariance indices must be unique')

    with h5py.File(filename, 'r') as h5:
        _read_representation(h5, filename)
        mode = h5.attrs.get('mode', '')
        if isinstance(mode, bytes):
            mode = mode.decode('utf-8')

        observation_count = components * target_time.size
        if mode == 'diagonal':
            diagonal_value = float(h5.attrs['diagonal_value'])
            if not np.isfinite(diagonal_value) or diagonal_value <= 0.0:
                raise ValueError(
                    'invalid diagonal covariance in {}'.format(filename)
                )
            return diagonal_value * np.eye(observation_count)

        if mode != 'stationary' or 'autocovariance' not in h5:
            raise ValueError(
                'stationary autocovariance is missing from {}'.format(
                    filename
                )
            )

        sample_interval = float(h5.attrs['sample_interval'])
        time_origin = float(h5.attrs.get('time_origin', 0.0))
        epoch_mapping = h5.attrs.get('epoch_mapping', 'exact')
        if isinstance(epoch_mapping, bytes):
            epoch_mapping = epoch_mapping.decode('utf-8')
        autocovariance = h5['autocovariance'][:]

    if autocovariance.shape[0] != components:
        raise ValueError(
            '{} covariance components required, but {} stores {}'.format(
                components, filename, autocovariance.shape[0]
            )
        )
    if target_time.size == 0:
        return np.empty((0, 0), dtype=float)

    if matrix_epoch_index is None:
        # Stationarity constrains time differences, not absolute epoch phase.
        # Different KIN arcs may start at :00, :10, etc., while every within-arc
        # separation is an exact multiple of the covariance sample interval.
        reference_epoch = target_time[0] if epoch_mapping == 'exact' else time_origin
        relative_epoch = (target_time - reference_epoch) / sample_interval
        epoch_index = np.rint(relative_epoch).astype(np.int64)
        alignment_error = np.abs(
            relative_epoch - epoch_index.astype(float)
        )
        if epoch_mapping == 'exact' and np.any(alignment_error > 1.0e-6):
            raise ValueError(
                'target epoch differences in {} are not aligned to its {} s covariance '
                'lag grid'.format(filename, sample_interval)
            )
        if epoch_mapping not in ('exact', 'nearest'):
            raise ValueError(
                'unsupported epoch mapping in {}: {!r}'.format(
                    filename, epoch_mapping
                )
            )
        if np.unique(epoch_index).size != epoch_index.size:
            raise ValueError(
                'multiple target epochs map to the same covariance-grid '
                'epoch in {}'.format(filename)
            )

    lag = np.abs(epoch_index[:, None] - epoch_index[None, :])
    if lag.max() >= autocovariance.shape[1]:
        raise ValueError(
            'requested lag {} exceeds the maximum stored lag {} in {}'.format(
                int(lag.max()), autocovariance.shape[1] - 1, filename
            ) + '; regenerate this covariance model with the current '
            'CalibrateOrbit.py'
        )

    if components == 1:
        covariance = autocovariance[0, lag]
    else:
        covariance = np.zeros(
            (3 * target_time.size, 3 * target_time.size), dtype=float
        )
        for axis in range(3):
            covariance[axis::3, axis::3] = autocovariance[axis, lag]

    covariance = 0.5 * (covariance + covariance.T)
    if not np.isfinite(covariance).all():
        raise ValueError(
            'materialized covariance contains NaN or infinity: {}'.format(
                filename
            )
        )
    return covariance


def combine_itsg_epoch_covariance(temporal_covariance, components):
    '''Combine ITSG epoch covariance with residual temporal correlation.

    The six ITSG columns are [xx, yy, zz, xy, xz, yz]. The ITSG blocks
    set absolute variance and simultaneous XYZ correlation. The posterior
    residual model supplies normalized temporal correlation.
    '''
    components = np.asarray(components, dtype=float)
    if components.ndim != 2 or components.shape[1] != 6:
        raise ValueError('ITSG covariance must have shape (epochs, 6)')
    if not np.isfinite(components).all():
        raise ValueError('ITSG covariance contains NaN or infinity')

    count = components.shape[0]
    temporal_covariance = np.asarray(temporal_covariance, dtype=float)
    shape = (3 * count, 3 * count)
    if temporal_covariance.shape != shape:
        raise ValueError(
            'temporal covariance has shape {}, expected {}'.format(
                temporal_covariance.shape, shape
            )
        )
    if not np.isfinite(temporal_covariance).all():
        raise ValueError('temporal covariance contains NaN or infinity')
    if count == 0:
        return np.empty((0, 0), dtype=float)

    blocks = np.empty((count, 3, 3), dtype=float)
    blocks[:, 0, 0] = components[:, 0]
    blocks[:, 1, 1] = components[:, 1]
    blocks[:, 2, 2] = components[:, 2]
    blocks[:, 0, 1] = blocks[:, 1, 0] = components[:, 3]
    blocks[:, 0, 2] = blocks[:, 2, 0] = components[:, 4]
    blocks[:, 1, 2] = blocks[:, 2, 1] = components[:, 5]
    try:
        factors = np.linalg.cholesky(blocks)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            'ITSG epoch covariance must be positive definite'
        ) from exc

    variance = np.diag(temporal_covariance)
    if np.any(variance <= 0):
        raise ValueError('temporal covariance diagonal must be positive')
    deviation = np.sqrt(variance)
    correlation = temporal_covariance / np.outer(deviation, deviation)
    correlation = 0.5 * (correlation + correlation.T)
    correlation_blocks = correlation.reshape(count, 3, count, 3)
    diagonal_blocks = correlation_blocks[
        np.arange(count), :, np.arange(count), :
    ]
    if not np.allclose(
            diagonal_blocks, np.eye(3)[None, :, :],
            rtol=1e-10, atol=1e-12):
        raise ValueError(
            'ITSGEpoch requires axis-wise FullEstimated temporal covariance'
        )

    # C = blockdiag(L_i) @ R @ blockdiag(L_i).T, S_i = L_i @ L_i.T.
    left = np.einsum(
        'iab,ibjc->iajc', factors, correlation_blocks, optimize=True
    )
    combined = np.einsum(
        'iajc,jdc->iajd', left, factors, optimize=True
    ).reshape(shape)
    return 0.5 * (combined + combined.T)
