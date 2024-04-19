# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2024/4/19 19:12
# @File    : demo_timeseries.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import src.Preference.EnumType as EnumType
from src.Interface.ArcOperation import ArcData
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
import json
import os
import pathlib
import h5py
import datetime
import calendar


def orbitseries():
    adjust_config = AdjustOrbitConfig()
    adjustOrbitDict = json.load(open('../setting/Calibrate/AdjustOrbitConfig.json', 'r'))
    adjust_config.__dict__ = adjustOrbitDict
    interface_config = InterfaceConfig()
    interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict


    AdjustPath = AdjustOrbitConfig.PathOfFiles()
    AdjustPath.__dict__.update(adjust_config.PathOfFilesConfig.copy())

    resultpath = '../result/orbit/'
    date_span = get_month_start_and_end_by_date_interval('2006-01-01', '2010-12-31')

    for i in range(len(date_span)):
        filename = "../temp/RawData/" + str(date_span[i][0]) + '_' + str(date_span[i][1]) + ".hdf5"
        h5data = h5py.File(filename, "r")
        resulth5 = h5py.File(resultpath + str(date_span[i][0]) + '_' + str(date_span[i][1]), "w")
        res_dir = "../temp/StateVector/" + str(date_span[i][0]) + '_' + str(date_span[i][1])
        RMSAX = []
        RMSAY = []
        RMSAZ = []
        RMSBX = []
        RMSBY = []
        RMSBZ = []
        for j in range(0, 115):
            res_filename = res_dir + '/' + str(j) + '_AB.hdf5'
            h5 = h5py.File(res_filename, 'r')
            rAB = h5['r'][:]
            rA = rAB[:, 0:3, 0]
            rB = rAB[:, 3:6, 0]

            gnva = h5data['GNV_A_arc' + str(j)][()][:, 1:4]
            gnvb = h5data['GNV_B_arc' + str(j)][()][:, 1:4]

            if np.shape(gnva) == np.shape(rA):
                diff = gnva - rA
                RMSAX.append(__rms(diff[:, 0]))
                RMSAY.append(__rms(diff[:, 1]))
                RMSAZ.append(__rms(diff[:, 2]))

            if np.shape(gnvb) == np.shape(rB):
                diff = gnvb - rB
                RMSBX.append(__rms(diff[:, 0]))
                RMSBY.append(__rms(diff[:, 1]))
                RMSBZ.append(__rms(diff[:, 2]))

        h5data.close()
        resulth5.create_dataset('AX', data=RMSAX)
        resulth5.create_dataset('AY', data=RMSAY)
        resulth5.create_dataset('AZ', data=RMSAZ)
        resulth5.create_dataset('BX', data=RMSBX)
        resulth5.create_dataset('BY', data=RMSBY)
        resulth5.create_dataset('BZ', data=RMSBZ)
        resulth5.close()
    pass


def get_month_start_and_end_by_date_interval(start_date, end_date):
    """
    获取指定时间段内每个月的起止时间参数列表
    :param start_date: 起始时间 --> str
    :param end_date: 结束时间 --> str
    :return: date_range_list -->list [{'2019-10-01': ['2019-10-01', '2019-10-31']}, ...}]

    """
    date_range_list = []
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    # 计算开始日期的当月起止日期
    start_month_first = datetime.datetime(year=start_date.year, month=start_date.month, day=1).date()
    start_month_last = datetime.datetime(year=start_date.year, month=start_date.month,
                                         day=calendar.monthrange(start_date.year, start_date.month)[1]).date()
    date_range_list.append([start_month_first.strftime('%Y-%m-%d'),start_month_last.strftime('%Y-%m-%d')])

    # 计算结束日期的当月起止日期
    end_month_first = datetime.datetime(year=end_date.year, month=end_date.month, day=1).date()
    end_month_last = datetime.datetime(year=end_date.year, month=end_date.month,
                                       day=calendar.monthrange(end_date.year, end_date.month)[1]).date()

    while True:
        # 下个月开始日期
        next_month_start = datetime.datetime(year=start_date.year, month=start_date.month,
                                             day=calendar.monthrange(start_date.year, start_date.month)[
                                                 1]).date() + datetime.timedelta(days=1)
        next_month_end = datetime.datetime(year=next_month_start.year, month=next_month_start.month,
                                           day=calendar.monthrange(next_month_start.year, next_month_start.month)[
                                               1]).date()
        if next_month_start < end_month_first:
            date_range_list.append([next_month_start.strftime('%Y-%m-%d'),next_month_end.strftime('%Y-%m-%d')])
            start_date = next_month_start
        else:
            break
    # 避免传入的起止日期在同一月导致数据重复
    temp_dict = [end_month_first.strftime('%Y-%m-%d'), end_month_last.strftime('%Y-%m-%d')]
    if temp_dict not in date_range_list:
        date_range_list.append(temp_dict)
    return date_range_list


def __rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


def kbrrseries():
    pass


if __name__ == '__main__':
    orbitseries()