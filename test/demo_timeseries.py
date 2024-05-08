# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2024/4/19 19:12
# @File    : demo_timeseries.py
# @Software: PyCharm
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import matplotlib
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
        if not os.path.exists(filename):
            continue
        h5data = h5py.File(filename, "r")
        resulth5 = h5py.File(resultpath + str(date_span[i][0]) + '_' + str(date_span[i][1]) + '.hdf5', "w")
        res_dir = "../temp/StateVector/" + str(date_span[i][0]) + '_' + str(date_span[i][1])
        RMSAX = []
        RMSAY = []
        RMSAZ = []
        RMSBX = []
        RMSBY = []
        RMSBZ = []
        for j in range(0, 115):
            res_filename = res_dir + '/' + str(j) + '_AB.hdf5'
            if not os.path.exists(res_filename):
                break
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


def draworbit():
    date_span = get_month_start_and_end_by_date_interval('2006-01-01', '2006-12-31')
    filepath = '../result/orbit/'
    rmsax = []
    rmsay = []
    rmsaz = []
    rmsbx = []
    rmsby = []
    rmsbz = []
    for i in range(len(date_span)):
        filename = filepath + str(date_span[i][0]) + '_' +str(date_span[i][1]) + '.hdf5'
        h5data = h5py.File(filename, 'r')
        rmsax = np.append(rmsax, outliner(h5data['AX'][:]))
        rmsay = np.append(rmsay, outliner(h5data['AY'][:]))
        rmsaz = np.append(rmsaz, outliner(h5data['AZ'][:]))
        rmsbx = np.append(rmsbx, outliner(h5data['BX'][:]))
        rmsby = np.append(rmsby, outliner(h5data['BY'][:]))
        rmsbz = np.append(rmsbz, outliner(h5data['BZ'][:]))

    matplotlib.rcParams.update({'font.size': 30})

    plt.style.use(['science', 'grid'])

    namelist = ['2006/02','2006/05','2006/08','2006/11']
    # plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(3, 2, figsize=(15, 9), sharex=True, sharey=True)

    fig.subplots_adjust(left=0.066, right=0.981, top=0.923, bottom=0.08, wspace=0.1, hspace=0.15)
    ax[0, 0].set_title('GRACE A')
    ax[0, 0].plot(np.arange(len(rmsax)), rmsax * 1000, color='black')
    ax[0, 0].axhline(y=5.8, color='#ED4C67', linestyle='--', linewidth=3)
    ax[0, 0].arrow(650, 20, 0, -9, linewidth=2, color='#ED4C67', linestyle='--', head_width=20, head_length=3, fc='blue', ec='blue')
    ax[0, 0].text(550, 20, '5.8mm')
    ax[1, 0].plot(np.arange(len(rmsay)), rmsay * 1000, color='black')
    ax[1, 0].axhline(y=5.4, color='#ED4C67', linestyle='--', linewidth=3)
    ax[2, 0].plot(np.arange(len(rmsaz)), rmsaz * 1000, color='black')
    ax[2, 0].axhline(y=5.3, color='#ED4C67', linestyle='--', linewidth=3)
    ax[0, 0].set(xlim=[np.arange(len(rmsax))[0], np.arange(len(rmsax))[-1]], ylim=[0, 30], ylabel='X [mm]')
    ax[1, 0].set(xlim=[np.arange(len(rmsax))[0], np.arange(len(rmsax))[-1]], ylim=[0, 30], ylabel='Y [mm]')
    ax[2, 0].set(xlim=[np.arange(len(rmsax))[0], np.arange(len(rmsax))[-1]], ylim=[0, 30], ylabel='Z [mm]')
    ax[2, 0].set_xticks([150, 500, 850, 1200])
    ax[2, 0].set_xticklabels(namelist)
    ax[0, 1].set_title('GRACE B')
    ax[0, 1].plot(np.arange(len(rmsbx)), rmsbx * 1000, color='black')
    ax[0, 1].axhline(y=5.7, color='#ED4C67', linestyle='--', linewidth=3)
    ax[1, 1].plot(np.arange(len(rmsby)), rmsby * 1000, color='black')
    ax[1, 1].axhline(y=5.6, color='#ED4C67', linestyle='--', linewidth=3)
    ax[2, 1].plot(np.arange(len(rmsbz)), rmsbz * 1000, color='black')
    ax[2, 1].axhline(y=5.5, color='#ED4C67', linestyle='--', linewidth=3)
    ax[0, 1].set(xlim=[np.arange(len(rmsbx))[0], np.arange(len(rmsbx))[-1]])
    ax[1, 1].set(xlim=[np.arange(len(rmsbx))[0], np.arange(len(rmsbx))[-1]])
    ax[2, 1].set(xlim=[np.arange(len(rmsbx))[0], np.arange(len(rmsbx))[-1]])
    # ax3.set_xticks(np.arange(11))
    ax[2, 1].set_xticklabels(namelist)

    plt.show()


def outliner(data):
    temp = np.array(data)
    index = [temp < 0.03]
    data = temp[index]
    return list(data)


def kbrrseries():
    adjust_config = AdjustOrbitConfig()
    adjustOrbitDict = json.load(open('../setting/Calibrate/AdjustOrbitConfig.json', 'r'))
    adjust_config.__dict__ = adjustOrbitDict
    interface_config = InterfaceConfig()
    interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict
    AdjustPath = AdjustOrbitConfig.PathOfFiles()
    AdjustPath.__dict__.update(adjust_config.PathOfFilesConfig.copy())

    resultpath = '../result/kbrr/'
    date_span = get_month_start_and_end_by_date_interval('2006-01-01', '2010-12-31')

    for i in range(len(date_span)):
        post_RMS = []
        res_dir = "../temp/RangeRate/" + str(date_span[i][0]) + '_' + str(date_span[i][1])

        resulth5 = h5py.File(resultpath + str(date_span[i][0]) + '_' + str(date_span[i][1]) + '.hdf5', "w")
        for j in range(0, 115):
            res_filename = res_dir + '/RangeRate_' + str(j) + '.hdf5'
            if not os.path.exists(res_filename):
                continue
            h5 = h5py.File(res_filename, 'r')
            if len(h5.keys()) != 0:
                postfit = h5['post_residual'][()]
                post_RMS.append(__rms(postfit))
                h5.close()
                print(__rms(postfit))
        print(date_span[i][0])
        resulth5.create_dataset('rms', data=post_RMS)
        resulth5.close()
    pass


def drawkbrr():
    date_span = get_month_start_and_end_by_date_interval('2006-01-01', '2010-11-30')
    filepath = '../result/kbrr/'
    rms = []
    for i in range(len(date_span)):
        filename = filepath + str(date_span[i][0]) + '_' +str(date_span[i][1]) + '.hdf5'
        h5data = h5py.File(filename, 'r')
        rms = np.append(rms, h5data['rms'][:])
    matplotlib.rcParams.update({'font.size': 30})

    plt.style.use(['science', 'grid'])

    plt.style.use(['grid'])
    namelist = ['2006/12','2007/12','2008/12', '2009/12', '2010/12']
    plt.scatter(np.arange(len(rms)), rms * 1e6, s=20, color='black')
    plt.xlim([np.arange(len(rms))[0], np.arange(len(rms))[-1]])
    plt.ylim([0.1, 0.5])
    plt.ylabel('Residual [um/s]')
    plt.xticks([1354, 2708, 4062, 5416, 6770], namelist)

    # plt.set(xlim=, ylim=[0, 30], ylabel='X [mm]')
    plt.show()
    pass


if __name__ == '__main__':
    # orbitseries()
    # draworbit()
    # kbrrseries()
    drawkbrr()