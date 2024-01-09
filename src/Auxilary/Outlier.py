"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/3/25 上午11:17
@Description:
"""
import numpy as np
from src.Auxilary.GeoMathKit import GeoMathKit


class Outlier:

    def __init__(self, rms_times, upper_RMS=None, upper_obs=None):
        """
        :param rms_times: value greater than std_times*std will be treated as an outlier
        :param upper_RMS: RMS greater than given upper_RMS ? Remove the whole data
        :param upper_obs: value greater than upper_obs will be treated as an outlier
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

        leng = np.shape(t)[0]
        index1 = list(np.ones(leng).astype(bool))
        index2 = list(np.ones(leng).astype(bool))
        index3 = list(np.ones(leng).astype(bool))

        rms = GeoMathKit.rms(data)
        if self.__upper_RMS is not None and rms > self.__upper_RMS:
            index1 = list(np.zeros(leng).astype(bool))

        index2 = data < rms * self.__rms_times

        if self.__upper_obs is not None:
            index3 = data < self.__upper_obs

        '''index of the inliers'''
        index = np.array(index1) * np.array(index2) * np.array(index3)

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