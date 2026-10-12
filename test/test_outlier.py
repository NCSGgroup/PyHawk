import numpy as np

from src.Auxilary.Outlier import ArcOutlier, EpochOutlier, Outlier


def test_kbrr_outlier_filter_is_robust_symmetric_and_keeps_the_arc():
    data = 100e-6 + np.array(
        [-50.0, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 50.0]
    ) * 1e-6

    index = Outlier(
        rms_times=3,
        upper_RMS=60e-6,
        upper_obs=3
    ).remove(np.arange(data.size), data)

    assert index.tolist() == [
        False, True, True, True, True, True, True, True, False
    ]


def test_arc_outlier_detects_relative_monthly_anomalies():
    phase = np.linspace(0, 4 * np.pi, 126, endpoint=False)
    normal = 0.24e-6 * np.exp(
        0.08 * np.sin(phase) + 0.04 * np.cos(2 * phase)
    )
    metric = np.r_[
        normal,
        np.sqrt(0.463e-6 * 0.516e-6),
        np.sqrt(0.556e-6 * 0.508e-6),
        np.sqrt(0.718e-6 * 0.746e-6)
    ]

    rejected, diagnostics = ArcOutlier(
        significance=0.01,
        max_fraction=0.1,
        min_samples=8
    ).detect(metric)

    assert np.flatnonzero(rejected).tolist() == [126, 127, 128]
    assert diagnostics['outlier_count'] == 3


def test_arc_outlier_is_scale_invariant():
    metric = np.r_[
        np.linspace(0.20e-6, 0.28e-6, 100),
        [0.47e-6, 0.56e-6, 0.72e-6]
    ]
    detector = ArcOutlier(
        significance=0.01,
        max_fraction=0.1,
        min_samples=8
    )

    rejected, _ = detector.detect(metric)
    scaled_rejected, _ = detector.detect(metric * 1000)

    np.testing.assert_array_equal(rejected, scaled_rejected)


def test_lower_tail_is_removed_before_upper_tail_screening():
    rng = np.random.default_rng(4)
    phase = np.linspace(0, 4 * np.pi, 126, endpoint=False)
    normal = 0.24e-6 * np.exp(
        0.08 * np.sin(phase) + 0.04 * np.cos(2 * phase)
    ) + rng.normal(0.0, 0.02e-6, 126)
    metric = np.r_[normal, 0.47e-6, 0.56e-6, 0.72e-6, 1e-42]
    detector = ArcOutlier(
        significance=0.01,
        max_fraction=0.1,
        min_samples=8
    )

    lower_rejected, _ = detector.detect(metric, tail='lower')
    assert np.flatnonzero(lower_rejected).tolist() == [129]

    retained = np.flatnonzero(~lower_rejected)
    upper_rejected, _ = detector.detect(metric[retained], tail='upper')
    assert retained[np.flatnonzero(upper_rejected)].tolist() == [126, 127, 128]


def test_epoch_outlier_detects_a_local_spike_without_rejecting_quiet_epochs():
    rng = np.random.default_rng(20260311)
    time = np.arange(200, dtype=float) * 5.0
    residual = rng.normal(0.0, 0.15e-6, time.size)
    residual[100] = 8.0e-6

    keep, diagnostics = EpochOutlier(
        significance=0.01,
        max_fraction=0.02,
        min_samples=20
    ).detect(time, residual)

    assert not keep[100]
    assert diagnostics['outlier_count'] >= 1
    assert np.count_nonzero(~keep) <= 3


def test_epoch_outlier_leaves_dense_departures_for_arc_level_qc():
    rng = np.random.default_rng(31)
    time = np.arange(200, dtype=float) * 5.0
    residual = rng.normal(0.0, 0.05e-6, time.size)
    residual[20:80:4] = 8.0e-6

    keep, diagnostics = EpochOutlier(
        significance=0.01,
        max_fraction=0.02,
        min_samples=20
    ).detect(time, residual)

    assert np.all(keep)
    assert diagnostics['outlier_count'] == 0


if __name__ == '__main__':
    test_kbrr_outlier_filter_is_robust_symmetric_and_keeps_the_arc()
    test_arc_outlier_detects_relative_monthly_anomalies()
    test_arc_outlier_is_scale_invariant()
    test_lower_tail_is_removed_before_upper_tail_screening()
    test_epoch_outlier_detects_a_local_spike_without_rejecting_quiet_epochs()
    test_epoch_outlier_leaves_dense_departures_for_arc_level_qc()
    print('test_outlier: PASS')
