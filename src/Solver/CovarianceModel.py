import os

import h5py
import numpy as np


REPRESENTATION = 'stationary_autocovariance_v1'


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


def load_covariance(filename, target_time, components):
    """Materialize the covariance at the requested epochs.

    The lag of each matrix element is computed from the physical epoch
    difference, so missing observations do not compress the covariance lag.
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

    relative_epoch = (target_time - time_origin) / sample_interval
    epoch_index = np.rint(relative_epoch).astype(np.int64)
    alignment_error = np.abs(
        relative_epoch - epoch_index.astype(float)
    )
    if epoch_mapping == 'exact' and np.any(alignment_error > 1.0e-6):
        raise ValueError(
            'target epochs in {} are not aligned to its {} s covariance '
            'grid'.format(filename, sample_interval)
        )
    if epoch_mapping not in ('exact', 'nearest'):
        raise ValueError(
            'unsupported epoch mapping in {}: {!r}'.format(
                filename, epoch_mapping
            )
        )
    if np.unique(epoch_index).size != epoch_index.size:
        raise ValueError(
            'multiple target epochs map to the same covariance-grid epoch '
            'in {}'.format(filename)
        )

    lag = np.abs(epoch_index[:, None] - epoch_index[None, :])
    if lag.max() >= autocovariance.shape[1]:
        raise ValueError(
            'requested lag {} exceeds the maximum stored lag {} in {}'.format(
                int(lag.max()), autocovariance.shape[1] - 1, filename
            )
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
