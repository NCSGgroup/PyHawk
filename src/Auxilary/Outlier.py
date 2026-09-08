"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/3/25 上午11:17
@Description:
"""
import numpy as np
from scipy.stats import t as student_t
from src.Auxilary.GeoMathKit import GeoMathKit


class Outlier:

    def __init__(self, rms_times, upper_RMS=None, upper_obs=None):
        """
        :param rms_times: robust-sigma multiplier used to identify an outlier
        :param upper_RMS: absolute residual limit; no longer removes a whole arc
        :param upper_obs: optional second robust-sigma multiplier
        """
        assert rms_times > 0
        self.__rms_times = rms_times
        self.__upper_RMS = upper_RMS
        self.__upper_obs = upper_obs
        pass

    def remove(self, t: np.ndarray, data: np.ndarray):
        """

        :param t: epoch of the time-series [1-dim]
        :param data: data to be filtered [1-dim]
        :return:
        """
        data = np.asarray(data)
        index = np.isfinite(data)
        if not np.any(index):
            return index

        center = np.median(data[index])
        deviation = np.abs(data - center)
        scale = 1.4826 * np.median(deviation[index])

        if scale > 0:
            index &= deviation <= scale * self.__rms_times
            if self.__upper_obs is not None:
                index &= deviation <= scale * self.__upper_obs
        else:
            index &= deviation == 0

        if self.__upper_RMS is not None:
            index &= deviation <= self.__upper_RMS

        return index

    def remove_V2(self, t: np.ndarray, data: np.ndarray):
        """

        :param t: epoch of the time-series [1-dim]
        :param data: data to be filtered [1-dim]
        :return:
        """

        leng = np.shape(t)[0]
        index1 = list(np.ones(leng).astype(bool))
        index2 = list(np.ones(leng).astype(bool))
        index3 = list(np.ones(leng).astype(bool))

        rms = GeoMathKit.rms(data)
        if self.__upper_RMS is not None:
            index1 = np.abs(data) < self.__upper_RMS

        index2 = np.abs(data) < np.abs(rms) * self.__rms_times

        if self.__upper_obs is not None:
            index3 = data < self.__upper_obs

        '''index of the inliers'''
        index = np.array(index1) * np.array(index2) * np.array(index3)

        return index
    def remove_V3(self, t: np.ndarray, data: np.ndarray):
        """

        :param t: epoch of the time-series [1-dim]
        :param data: data to be filtered [1-dim]
        :return:
        """

        leng = np.shape(t)[0]
        index1 = list(np.ones(leng).astype(bool))
        index2 = list(np.ones(leng).astype(bool))
        index3 = list(np.ones(leng).astype(bool))

        rms = GeoMathKit.rms(data)
        if self.__upper_RMS is not None:
            index1 = np.abs(data) < self.__upper_RMS

        index2 = np.abs(data) < np.abs(rms) * self.__rms_times

        if self.__upper_obs is not None:
            index3 = data < self.__upper_obs

        '''index of the inliers'''
        index = np.array(index1) * np.array(index2) * np.array(index3)

        return index


class ArcOutlier:
    """Detect relative upper or lower outliers without an absolute limit."""

    def __init__(self, significance=0.01, max_fraction=0.1,
                 min_samples=8):
        if not 0 < significance < 1:
            raise ValueError('significance must be between zero and one')
        if not 0 < max_fraction < 0.5:
            raise ValueError('max_fraction must be between zero and 0.5')
        if min_samples < 3:
            raise ValueError('min_samples must be at least three')
        self.significance = float(significance)
        self.max_fraction = float(max_fraction)
        self.min_samples = int(min_samples)

    def detect(self, metric, tail='upper'):
        """Apply a robust one-sided generalized ESD test to log metrics.

        The logarithm makes the test scale-invariant, so multiplying all
        residuals by the same factor does not change the rejected arcs.  The
        ESD statistic uses the median and MAD at every removal step, so a
        simultaneous low-tail failure cannot inflate the scale and hide a
        high-tail failure.
        ``tail`` selects the statistically high (``upper``) or unusually
        small (``lower``) metrics.  A near-zero arc metric is not necessarily
        good data: it can result from an almost empty post-fit arc.
        """
        metric = np.asarray(metric, dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(metric)) or np.any(metric <= 0):
            raise ValueError('arc quality metrics must be finite and positive')
        if tail not in ('upper', 'lower'):
            raise ValueError("tail must be 'upper' or 'lower'")

        size = metric.size
        rejected = np.zeros(size, dtype=bool)
        diagnostics = {
            'outlier_count': 0,
            'candidate_indices': np.zeros(0, dtype=np.int64),
            'statistics': np.zeros(0, dtype=np.float64),
            'critical_values': np.zeros(0, dtype=np.float64)
        }
        if size < self.min_samples:
            return rejected, diagnostics

        values = np.log(metric)
        remaining = list(range(size))
        candidate_indices = []
        statistics = []
        critical_values = []
        max_outliers = min(
            max(1, int(np.floor(size * self.max_fraction))),
            size - 2
        )

        for _ in range(max_outliers):
            sample = values[remaining]
            sample_size = sample.size
            sample_center = np.median(sample)
            sample_scale = 1.4826 * np.median(
                np.abs(sample - sample_center)
            )
            if not np.isfinite(sample_scale) or sample_scale == 0:
                break

            if tail == 'upper':
                candidate_position = int(np.argmax(sample))
                statistic = (
                    sample[candidate_position] - sample_center
                ) / sample_scale
            else:
                candidate_position = int(np.argmin(sample))
                statistic = (
                    sample_center - sample[candidate_position]
                ) / sample_scale
            probability = 1.0 - self.significance / sample_size
            critical_t = student_t.ppf(probability, sample_size - 2)
            critical_value = (
                (sample_size - 1) * critical_t
                / np.sqrt(
                    sample_size
                    * (sample_size - 2 + critical_t ** 2)
                )
            )

            candidate_indices.append(remaining[candidate_position])
            statistics.append(float(statistic))
            critical_values.append(float(critical_value))
            remaining.pop(candidate_position)

        outlier_count = 0
        for index, (statistic, critical_value) in enumerate(
                zip(statistics, critical_values)):
            if statistic > critical_value:
                outlier_count = index + 1

        if outlier_count:
            rejected[np.asarray(
                candidate_indices[:outlier_count], dtype=np.int64
            )] = True

        diagnostics = {
            'outlier_count': outlier_count,
            'candidate_indices': np.asarray(
                candidate_indices, dtype=np.int64
            ),
            'statistics': np.asarray(statistics, dtype=np.float64),
            'critical_values': np.asarray(
                critical_values, dtype=np.float64
            )
        }
        return rejected, diagnostics


class EpochOutlier:
    """Identify isolated KBRR epochs with robust two-sided ESD tests.

    The tested series is the deviation from a local four-neighbour median.
    A post-fit KBRR residual may legitimately contain a smooth low-frequency
    signal, so testing its absolute amplitude would falsely label that signal
    as a collection of bad epochs.  A local innovation instead identifies
    only abrupt, isolated departures.  Its threshold is a Bonferroni-adjusted
    robust ESD critical value and therefore contains no absolute KBRR limit.
    """

    def __init__(self, significance=0.01, max_fraction=0.02,
                 min_samples=20):
        if not 0 < significance < 1:
            raise ValueError('significance must be between zero and one')
        if not 0 < max_fraction < 0.5:
            raise ValueError('max_fraction must be between zero and 0.5')
        if min_samples < 5:
            raise ValueError('min_samples must be at least five')
        self.significance = float(significance)
        self.max_fraction = float(max_fraction)
        self.min_samples = int(min_samples)

    def _detect_series(self, value):
        """Return one-pass robust two-sided ESD candidates in *value*."""
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        rejected = np.zeros(value.size, dtype=bool)
        diagnostics = {
            'outlier_count': 0,
            'candidate_indices': np.zeros(0, dtype=np.int64),
            'statistics': np.zeros(0, dtype=np.float64),
            'critical_values': np.zeros(0, dtype=np.float64)
        }
        if value.size < self.min_samples or not np.all(np.isfinite(value)):
            return rejected, diagnostics

        sample_size = value.size
        center = float(np.median(value))
        scale = float(1.4826 * np.median(np.abs(value - center)))
        if not np.isfinite(scale) or scale == 0.0:
            return rejected, diagnostics
        probability = 1.0 - self.significance / (2.0 * sample_size)
        critical_t = student_t.ppf(probability, sample_size - 2)
        critical_value = (
            (sample_size - 1) * critical_t
            / np.sqrt(
                sample_size
                * (sample_size - 2 + critical_t ** 2)
            )
        )
        statistics = np.abs(value - center) / scale
        candidates = np.flatnonzero(statistics > critical_value)
        max_outliers = max(1, int(np.floor(
            sample_size * self.max_fraction
        )))
        # A dense set of departures is a degraded arc, not a few removable
        # epochs.  Leave it for the arc-level quality test.
        if candidates.size <= max_outliers:
            rejected[candidates] = True
        else:
            candidates = np.zeros(0, dtype=np.int64)

        diagnostics = {
            'outlier_count': int(np.count_nonzero(rejected)),
            'candidate_indices': np.asarray(candidates, dtype=np.int64),
            'statistics': np.asarray(statistics, dtype=np.float64),
            'critical_values': np.full(
                statistics.size, critical_value, dtype=np.float64
            )
        }
        return rejected, diagnostics

    @staticmethod
    def _local_innovation(time, residual):
        """Return centre-minus-neighbour-median innovations on valid spans."""
        time = np.asarray(time, dtype=np.float64).reshape(-1)
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        if time.size != residual.size:
            raise ValueError('time and residual must have equal length')

        index = np.arange(time.size, dtype=np.int64)
        if time.size < 5:
            return index[:0], residual[:0]
        spacing = np.diff(time)
        positive_spacing = spacing[spacing > 0]
        if positive_spacing.size == 0:
            return index[:0], residual[:0]
        nominal_spacing = float(np.median(positive_spacing))
        contiguous = spacing <= 1.5 * nominal_spacing

        neighbours = np.column_stack((
            residual[:-4], residual[1:-3], residual[3:-1], residual[4:]
        ))
        valid = (
            np.convolve(
                contiguous.astype(np.int8), np.ones(4, dtype=np.int8),
                mode='valid'
            ) == 4
        )
        innovations = residual[2:-2] - np.median(neighbours, axis=1)
        valid &= np.isfinite(innovations) & np.all(
            np.isfinite(neighbours), axis=1
        )
        return (
            index[2:-2][valid],
            innovations[valid]
        )

    def detect(self, time, residual):
        """Return a keep mask and diagnostics for a post-fit KBRR arc."""
        time = np.asarray(time, dtype=np.float64).reshape(-1)
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        if time.size != residual.size:
            raise ValueError('time and residual must have equal length')

        keep = np.isfinite(time) & np.isfinite(residual)
        local_index, local_innovation = self._local_innovation(time, residual)
        local_rejected, local_diagnostics = self._detect_series(
            local_innovation
        )
        if local_rejected.size:
            keep[local_index[local_rejected]] = False

        return keep, {
            'outlier_count': int(np.count_nonzero(~keep)),
            'global_outlier_count': 0,
            'local_outlier_count': int(np.count_nonzero(local_rejected)),
            'global_diagnostics': None,
            'local_diagnostics': local_diagnostics
        }
