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
    """Detect month-relative upper outliers without an absolute RMS limit."""

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

    def detect(self, metric):
        """Apply a one-sided generalized ESD test to log arc metrics.

        The logarithm makes the test scale-invariant, so multiplying all
        monthly residuals by the same factor does not change the rejected
        arcs.  The returned mask is True only for statistically high arcs.
        """
        metric = np.asarray(metric, dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(metric)) or np.any(metric <= 0):
            raise ValueError('arc quality metrics must be finite and positive')

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
            sample_std = np.std(sample, ddof=1)
            if not np.isfinite(sample_std) or sample_std == 0:
                break

            candidate_position = int(np.argmax(sample))
            statistic = (
                sample[candidate_position] - np.mean(sample)
            ) / sample_std
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
