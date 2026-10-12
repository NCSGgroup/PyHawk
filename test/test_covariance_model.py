# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

import h5py
import numpy as np

from src.Solver.CovarianceModel import (
    covariance_matrix_epoch_index,
    load_covariance,
    normalize_covariance_epoch_mapping,
    save_stationary_covariance,
)


class TestStationaryCovarianceModel(unittest.TestCase):
    def test_epoch_mapping_selection_is_explicit(self):
        legacy_index = np.array([0, 12, 24])
        self.assertEqual(
            normalize_covariance_epoch_mapping('legacy_index'),
            'LegacyIndex'
        )
        self.assertEqual(
            normalize_covariance_epoch_mapping('physical_time'),
            'PhysicalTime'
        )
        self.assertIs(
            covariance_matrix_epoch_index(
                'LegacyIndex', legacy_index
            ),
            legacy_index
        )
        self.assertIsNone(
            covariance_matrix_epoch_index(
                'PhysicalTime', legacy_index
            )
        )

    def test_gap_uses_physical_time_lag(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'covariance.hdf5')
            autocovariance = np.array([4.0, 2.0, 1.0, 0.5])
            save_stationary_covariance(
                filename,
                autocovariance=autocovariance,
                sample_interval=5.0
            )

            covariance = load_covariance(
                filename,
                target_time=np.array([100.0, 105.0, 115.0]),
                components=1
            )

            np.testing.assert_allclose(
                covariance,
                np.array([
                    [4.0, 2.0, 0.5],
                    [2.0, 4.0, 1.0],
                    [0.5, 1.0, 4.0],
                ])
            )

    def test_three_component_model_is_interleaved(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'orbit_covariance.hdf5')
            autocovariance = np.array([
                [4.0, 1.0],
                [9.0, 2.0],
                [16.0, 3.0],
            ])
            save_stationary_covariance(
                filename,
                autocovariance=autocovariance,
                sample_interval=5.0
            )

            covariance = load_covariance(
                filename,
                target_time=np.array([0.0, 5.0]),
                components=3
            )

            np.testing.assert_allclose(
                np.diag(covariance), [4.0, 9.0, 16.0] * 2
            )
            self.assertEqual(covariance[0, 3], 1.0)
            self.assertEqual(covariance[1, 4], 2.0)
            self.assertEqual(covariance[2, 5], 3.0)
            self.assertEqual(covariance[0, 4], 0.0)

    def test_diagonal_model_is_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'diagonal.hdf5')
            save_stationary_covariance(
                filename, diagonal_value=0.004
            )

            with h5py.File(filename, 'r') as h5:
                self.assertNotIn('Sigma', h5)
                self.assertEqual(h5.attrs['mode'], 'diagonal')

            covariance = load_covariance(
                filename,
                target_time=np.array([0.0, 60.0]),
                components=3
            )
            np.testing.assert_allclose(covariance, 0.004 * np.eye(6))

    def test_missing_model_fails_instead_of_using_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'missing.hdf5')
            with self.assertRaises(FileNotFoundError):
                load_covariance(
                    filename, target_time=np.array([0.0]), components=1
                )

    def test_off_grid_epochs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'covariance.hdf5')
            save_stationary_covariance(
                filename,
                autocovariance=np.ones(4),
                sample_interval=5.0
            )
            with self.assertRaisesRegex(ValueError, 'not aligned'):
                load_covariance(
                    filename,
                    target_time=np.array([0.0, 6.0]),
                    components=1
                )

    def test_exact_stationary_covariance_is_invariant_to_arc_start_phase(self):
        # Job 254288: this KIN arc starts 10 seconds off the monthly origin,
        # but its internal lags are valid multiples of 60 seconds.
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'phase.h5')
            for components in (1, 3):
                acov = np.array([(axis + 1.) * .8 ** np.arange(5)
                                 for axis in range(components)])
                save_stationary_covariance(filename, autocovariance=acov,
                    sample_interval=60., time_origin=420638400.)
                expected = load_covariance(filename, [420638400., 420638460., 420638580.], components)
                for start in (421065190., 421065200., 421065215.5):
                    actual = load_covariance(filename, start + np.array([0.,60.,180.]), components)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
                    np.linalg.cholesky(actual)
                # Missing epochs preserve lag 3 rather than compressing to 2.
                self.assertEqual(actual[0, 2*components], acov[0,3])

    def test_phase_shift_does_not_allow_invalid_within_arc_lags(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'phase.h5')
            save_stationary_covariance(filename, autocovariance=.8**np.arange(5),
                sample_interval=60., time_origin=420638400.)
            with self.assertRaisesRegex(ValueError, 'epoch differences.*not aligned'):
                load_covariance(filename, [421065190.,421065255.], 1)

    def test_nearest_grid_mapping_remains_a_principal_submatrix(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'orbit_covariance.hdf5')
            autocovariance = 0.8 ** np.arange(32)
            save_stationary_covariance(
                filename,
                autocovariance=autocovariance,
                sample_interval=8.0,
                time_origin=0.0,
                allow_nearest_epoch=True
            )

            # The 60 s observations map to the configured 8 s grid at
            # indices [0, 8, 15], rather than compressing to [0, 1, 2].
            covariance = load_covariance(
                filename,
                target_time=np.array([0.0, 60.0, 120.0]),
                components=1
            )
            grid_index = np.array([0, 8, 15])
            expected_lag = np.abs(
                grid_index[:, None] - grid_index[None, :]
            )
            np.testing.assert_allclose(
                covariance, autocovariance[expected_lag]
            )
            np.linalg.cholesky(covariance)

    def test_matrix_indices_reproduce_legacy_orbit_submatrix(self):
        matrix_index = np.array([0, 12, 24])
        target_time = np.array([0.0, 60.0, 120.0])

        for sample_interval in (5.0, 8.0, 10.0):
            with self.subTest(sample_interval=sample_interval), \
                    tempfile.TemporaryDirectory() as directory:
                filename = os.path.join(
                    directory, 'orbit_covariance.hdf5'
                )
                autocovariance = np.vstack([
                    0.91 ** np.arange(32),
                    2.0 * 0.92 ** np.arange(32),
                    3.0 * 0.93 ** np.arange(32),
                ])
                save_stationary_covariance(
                    filename,
                    autocovariance=autocovariance,
                    sample_interval=sample_interval,
                    allow_nearest_epoch=True
                )

                covariance = load_covariance(
                    filename,
                    target_time=target_time,
                    components=3,
                    matrix_epoch_index=matrix_index
                )

                full_lag = np.abs(
                    np.arange(25)[:, None] - np.arange(25)[None, :]
                )
                dense = np.zeros((75, 75))
                for axis in range(3):
                    dense[axis::3, axis::3] = autocovariance[
                        axis, full_lag
                    ]
                interleaved_index = (
                    3 * matrix_index[:, None] + np.arange(3)
                ).reshape(-1)
                expected = dense[np.ix_(
                    interleaved_index, interleaved_index
                )]

                np.testing.assert_allclose(covariance, expected)

    def test_matrix_indices_ignore_physical_sample_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(
                directory, 'orbit_covariance.hdf5'
            )
            autocovariance = np.arange(32.0).reshape(1, -1)
            save_stationary_covariance(
                filename,
                autocovariance=autocovariance,
                sample_interval=8.0,
                allow_nearest_epoch=True
            )

            covariance = load_covariance(
                filename,
                target_time=np.array([0.0, 60.0, 120.0]),
                components=1,
                matrix_epoch_index=np.array([0, 12, 24])
            )
            expected_lag = np.array([
                [0, 12, 24],
                [12, 0, 12],
                [24, 12, 0],
            ])
            np.testing.assert_allclose(
                covariance, autocovariance[0, expected_lag]
            )


if __name__ == '__main__':
    unittest.main()
