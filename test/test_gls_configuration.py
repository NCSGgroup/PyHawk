# -*- coding: utf-8 -*-
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.fft import dct, idct

from src.Preference.Pre_CalibrateOrbit import CalibrateOrbitConfig
from src.Preference.Pre_Solver import SolverConfig
from src.Solver.CalibrateOrbit import CalibrateOrbit


class TestGLSConfiguration(unittest.TestCase):
    def test_solver_defaults_preserve_current_covariance_logic(self):
        config = SolverConfig()
        self.assertEqual(config.CovarianceGapMode, 'InterpolatePSD')
        self.assertEqual(
            config.CovariancePositiveDefiniteMode, 'DCTClip'
        )
        self.assertEqual(
            config.CovarianceEpochMapping, 'LegacyIndex'
        )
        self.assertIsNone(config.CovarianceMaxLagSeconds)

    def test_default_configuration_uses_fixed_diagonal(self):
        config = CalibrateOrbitConfig.GLSIteration()
        self.assertEqual(config.OrbitCovarianceMode, 'FixedDiagonal')
        self.assertEqual(config.OrbitFixedDiagonal, 0.004)
        self.assertEqual(config.RandomSeed, 20240724)

    def test_covariance_mode_validation(self):
        self.assertEqual(
            CalibrateOrbit._normalize_orbit_covariance_mode('diagonal'),
            'FixedDiagonal'
        )
        self.assertEqual(
            CalibrateOrbit._normalize_orbit_covariance_mode('full'),
            'EstimatedFull'
        )
        with self.assertRaisesRegex(ValueError, 'OrbitCovarianceMode'):
            CalibrateOrbit._normalize_orbit_covariance_mode('unknown')

    def test_covariance_experiment_mode_validation(self):
        self.assertEqual(
            CalibrateOrbit._normalize_covariance_gap_mode('ignore'),
            'Ignore'
        )
        self.assertEqual(
            CalibrateOrbit._normalize_covariance_gap_mode('mask'),
            'Mask'
        )
        self.assertEqual(
            CalibrateOrbit._normalize_covariance_gap_mode(
                'interpolate_psd'
            ),
            'InterpolatePSD'
        )
        self.assertEqual(
            CalibrateOrbit._normalize_covariance_positive_definite_mode(
                'diagonal_loading'
            ),
            'DiagonalLoading'
        )
        self.assertEqual(
            CalibrateOrbit._normalize_covariance_positive_definite_mode(
                'dct_clip'
            ),
            'DCTClip'
        )
        self.assertEqual(
            CalibrateOrbit._normalize_covariance_epoch_mapping(
                'legacy_index'
            ),
            'LegacyIndex'
        )
        self.assertEqual(
            CalibrateOrbit._normalize_covariance_epoch_mapping(
                'physical_time'
            ),
            'PhysicalTime'
        )
        with self.assertRaisesRegex(ValueError, 'CovarianceGapMode'):
            CalibrateOrbit._normalize_covariance_gap_mode('unknown')
        with self.assertRaisesRegex(
                ValueError, 'CovariancePositiveDefiniteMode'):
            CalibrateOrbit._normalize_covariance_positive_definite_mode(
                'unknown'
            )
        with self.assertRaisesRegex(
                ValueError, 'CovarianceEpochMapping'):
            CalibrateOrbit._normalize_covariance_epoch_mapping('unknown')

    def test_missing_experiment_parameters_use_production_defaults(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace()
        )
        self.assertEqual(
            calibrator._covariance_gap_mode(), 'InterpolatePSD'
        )
        self.assertEqual(
            calibrator._covariance_positive_definite_mode(), 'DCTClip'
        )
        self.assertEqual(
            calibrator._covariance_epoch_mapping(), 'LegacyIndex'
        )

    def test_epoch_mapping_default_preserves_legacy_indices(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace()
        )
        legacy_index = np.array([0, 12, 24])
        self.assertIs(
            calibrator._covariance_matrix_epoch_index(legacy_index),
            legacy_index
        )

    def test_physical_time_mapping_omits_legacy_indices(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(CovarianceEpochMapping='PhysicalTime')
        )
        self.assertIsNone(
            calibrator._covariance_matrix_epoch_index(
                np.array([0, 12, 24])
            )
        )

    def test_fixed_orbit_covariance_uses_configured_diagonal(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        calibrator.lagNum = 2
        covariance = calibrator.getOrbitSigma2(
            time=np.arange(2), res=np.zeros((2, 3)),
            diagonal_value=0.004
        )
        np.testing.assert_array_equal(covariance, 0.004 * np.eye(6))

    def test_estimated_orbit_covariance_uses_orbit_frequency(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(OrbitFre=8)
        )
        calibrator.cal_toeplitz_cov = Mock(
            return_value=(np.ones((3, 4)), 8.0)
        )

        result = calibrator.getOrbitSigma(
            time=np.arange(4) * 5.0,
            res=np.zeros((4, 3)),
            return_autocovariance=True
        )

        self.assertEqual(result[1], 8.0)
        self.assertEqual(
            calibrator.cal_toeplitz_cov.call_args.kwargs['dt'], 8.0
        )

    def test_saved_kbrr_autocovariance_includes_endpoint_lag(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        calibrator.lagNum = 4
        setattr(
            calibrator,
            '_CalibrateOrbit__interface_config',
            SimpleNamespace(sr_target=5.0)
        )
        calibrator._estimate_covariance_autocovariance = Mock(
            return_value=np.arange(8.0)
        )
        calibrator._make_autocovariance_positive_definite = Mock(
            side_effect=lambda acovariance, floor_factor, **kwargs: acovariance
        )

        autocovariance, sample_interval = calibrator.getKBRRSigma(
            time=np.arange(3) * 5.0,
            res=np.array([1.0, 2.0, 4.0]),
            return_autocovariance=True
        )

        self.assertEqual(sample_interval, 5.0)
        np.testing.assert_array_equal(
            autocovariance, np.arange(5.0)
        )

    def test_saved_orbit_autocovariance_includes_endpoint_lag(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        calibrator.lagNum = 4
        calibrator._estimate_covariance_autocovariance = Mock(
            return_value=np.arange(8.0)
        )
        calibrator._make_autocovariance_positive_definite = Mock(
            side_effect=lambda acovariance, floor_factor, **kwargs: acovariance
        )

        autocovariance, sample_interval = calibrator.cal_toeplitz_cov(
            time=np.arange(3) * 5.0,
            res_x=np.ones(3),
            res_y=np.ones(3),
            res_z=np.ones(3),
            dt=5.0,
            return_autocovariance=True
        )

        self.assertEqual(sample_interval, 5.0)
        self.assertEqual(autocovariance.shape, (3, 5))
        np.testing.assert_array_equal(autocovariance[0], np.arange(5.0))

    def test_iteration_dispatches_covariance_mode(self):
        for mode, expected in (
                ('FixedDiagonal', 'GLS'),
                ('EstimatedFull', 'GLS_full')):
            calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
            calibrator.GLS = Mock()
            calibrator.GLS_full = Mock()
            calibrator._build_neq = Mock()
            setattr(
                calibrator, '_CalibrateOrbit__calculate_cs',
                Mock(return_value=np.zeros(1))
            )
            calibrator.AccelerometerConfig = SimpleNamespace(isScale=False)
            config = SimpleNamespace(
                MaxIteration=1,
                MinIteration=1,
                RelativeTolerance=1e-3,
                AbsoluteTolerance=1e-15,
                OrbitCovarianceMode=mode,
                OrbitFixedDiagonal=(
                    0.004 if mode == 'FixedDiagonal' else -1.0
                )
            )

            calibrator._iterative_gls(1, config)

            if expected == 'GLS':
                calibrator.GLS.assert_called_once_with(
                    orbit_fixed_diagonal=0.004
                )
                calibrator.GLS_full.assert_not_called()
            else:
                calibrator.GLS.assert_not_called()
                calibrator.GLS_full.assert_called_once_with()

    def test_colored_gap_noise_is_reproducible(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        frequencies = np.array([0.0, 0.1, 0.2])
        power = np.array([1.0, 0.5, 0.25])

        calibrator._gls_rng = np.random.default_rng(1234)
        first = calibrator.generate_colored_noise_from_psd(
            frequencies, power, N=16, dt=5.0
        )
        calibrator._gls_rng = np.random.default_rng(1234)
        second = calibrator.generate_colored_noise_from_psd(
            frequencies, power, N=16, dt=5.0
        )

        np.testing.assert_array_equal(first, second)

    def test_default_gap_mode_calls_existing_interpolation_path(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        calibrator.lagNum = 3
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(CovarianceGapMode='InterpolatePSD')
        )
        calibrator.fill_gaps_with_interp_and_psd = Mock(
            return_value=(
                np.array([0.0, 5.0, 10.0]),
                np.array([1.0, 2.0, 3.0])
            )
        )
        calibrator.estimate_acov_stable = Mock(
            return_value=np.array([2.0, 1.0, 0.5])
        )

        result = calibrator._estimate_covariance_autocovariance(
            time=np.array([0.0, 10.0]),
            residual=np.array([1.0, 3.0]),
            sample_interval=5.0
        )

        calibrator.fill_gaps_with_interp_and_psd.assert_called_once()
        estimate_args = calibrator.estimate_acov_stable.call_args
        np.testing.assert_array_equal(
            estimate_args.args[0], np.array([1.0, 2.0, 3.0])
        )
        self.assertEqual(estimate_args.kwargs['max_lag'], 2)
        np.testing.assert_array_equal(result, [2.0, 1.0, 0.5])

    def test_physical_max_lag_limits_interpolation_estimator(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        calibrator.lagNum = 4320
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(
                CovarianceGapMode='InterpolatePSD',
                CovarianceMaxLagSeconds=21600
            )
        )
        filled = np.arange(5000.0)
        calibrator.fill_gaps_with_interp_and_psd = Mock(
            return_value=(np.arange(filled.size) * 10.0, filled)
        )
        calibrator.estimate_acov_stable = Mock(
            return_value=np.ones(2161)
        )

        calibrator._estimate_covariance_autocovariance(
            time=np.array([0.0, 10.0]),
            residual=np.array([1.0, 2.0]),
            sample_interval=10.0
        )

        self.assertEqual(
            calibrator.estimate_acov_stable.call_args.kwargs['max_lag'],
            2160
        )

    def test_invalid_physical_max_lag_is_rejected(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(CovarianceMaxLagSeconds=0)
        )
        with self.assertRaisesRegex(
                ValueError, 'CovarianceMaxLagSeconds'):
            calibrator._covariance_max_lag(
                sample_interval=10.0,
                available_count=100,
                legacy_max_lag=99
            )

    def test_mask_gap_mode_uses_existing_gap_aware_estimator(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        calibrator.lagNum = 3
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(CovarianceGapMode='Mask')
        )
        calibrator.autocorr_with_gaps_fft = Mock(
            return_value=(
                np.array([0.0, 5.0, 10.0]),
                np.array([4.0, 2.0, 1.0]),
                np.array([3.0, 1.0, 1.0])
            )
        )

        result = calibrator._estimate_covariance_autocovariance(
            time=np.array([0.0, 5.0, 15.0]),
            residual=np.array([1.0, 2.0, 3.0]),
            sample_interval=5.0
        )

        calibrator.autocorr_with_gaps_fft.assert_called_once()
        self.assertFalse(
            calibrator.autocorr_with_gaps_fft.call_args.kwargs['normalize']
        )
        np.testing.assert_array_equal(result, [4.0, 2.0, 1.0])

    def test_mask_selects_observations_on_covariance_grid(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        calibrator.lagNum = 4
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(CovarianceGapMode='Mask')
        )
        calibrator.autocorr_with_gaps_fft = Mock(
            return_value=(
                np.array([0.0, 10.0, 20.0]),
                np.array([4.0, 2.0, 1.0]),
                np.array([3.0, 1.0, 1.0])
            )
        )

        calibrator._estimate_covariance_autocovariance(
            time=np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
            residual=np.arange(6.0),
            sample_interval=10.0
        )

        call = calibrator.autocorr_with_gaps_fft.call_args
        np.testing.assert_array_equal(
            call.args[0], np.array([0.0, 10.0, 20.0])
        )
        np.testing.assert_array_equal(
            call.args[1], np.array([0.0, 2.0, 4.0])
        )

    def test_ignore_selects_observations_on_covariance_grid(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        calibrator.lagNum = 4
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(CovarianceGapMode='Ignore')
        )
        calibrator.estimate_acov_stable = Mock(
            return_value=np.array([2.0, 1.0, 0.5])
        )

        calibrator._estimate_covariance_autocovariance(
            time=np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
            residual=np.arange(6.0),
            sample_interval=10.0
        )

        call = calibrator.estimate_acov_stable.call_args
        np.testing.assert_array_equal(
            call.args[0], np.array([0.0, 2.0, 4.0])
        )
        self.assertEqual(call.kwargs['max_lag'], 2)

    def test_gap_fft_allocates_from_rounded_last_grid_index(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        _, autocovariance, pair_count = calibrator.autocorr_with_gaps_fft(
            t=np.array([0.0, 10.0, 20.0]),
            r=np.array([1.0, 2.0, 3.0]),
            dt=10.0,
            max_lag=2,
            normalize=False
        )

        self.assertEqual(autocovariance.size, 3)
        self.assertEqual(pair_count.size, 3)
        self.assertTrue(np.all(np.isfinite(autocovariance)))

    def test_default_dct_clip_is_numerically_unchanged(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(
                CovariancePositiveDefiniteMode='DCTClip'
            )
        )
        autocovariance = np.array([4.0, 3.0, -1.0, 0.5])
        floor = max(autocovariance[0] * 1e-3, 1e-18)
        expected = idct(
            np.maximum(
                dct(autocovariance, type=1, norm=None), floor
            ),
            type=1,
            norm=None
        )

        actual = calibrator._make_autocovariance_positive_definite(
            autocovariance, floor_factor=1e-3
        )

        np.testing.assert_array_equal(actual, expected)

    def test_diagonal_loading_only_changes_lag_zero(self):
        calibrator = CalibrateOrbit.__new__(CalibrateOrbit)
        setattr(
            calibrator,
            '_CalibrateOrbit__solver_config',
            SimpleNamespace(
                CovariancePositiveDefiniteMode='DiagonalLoading'
            )
        )
        autocovariance = np.array([1.0, 2.0, 1.0, 0.5])
        corrected = calibrator._make_autocovariance_positive_definite(
            autocovariance, floor_factor=1e-3
        )

        np.testing.assert_array_equal(
            corrected[1:], autocovariance[1:]
        )
        self.assertGreaterEqual(
            np.min(dct(corrected, type=1, norm=None)),
            autocovariance[0] * 1e-3 - 1e-12
        )

    def test_kbrr_fit_cycles_uses_saved_one_cpr_phase(self):
        phase = np.linspace(0.0, 2.0 * np.pi * 1.08, 100)
        elapsed = np.linspace(0.0, 6000.0, phase.size)
        matrix = np.column_stack((
            np.ones(phase.size), elapsed, elapsed ** 2,
            np.cos(phase), elapsed * np.cos(phase),
            np.sin(phase), elapsed * np.sin(phase)
        ))

        cycles = CalibrateOrbit._kbrr_fit_cycles(
            matrix, 'biasC0C1C2_OneCPR'
        )

        self.assertAlmostEqual(cycles, 1.08, places=12)
        self.assertIsNone(
            CalibrateOrbit._kbrr_fit_cycles(matrix, 'biasC0C1C2')
        )


if __name__ == '__main__':
    unittest.main()
