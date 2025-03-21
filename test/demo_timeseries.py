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
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
import json
import os
import pathlib
import h5py
import datetime
import calendar
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.Pre_Frame import FrameConfig
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType


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
    config = {
        "font.family": 'serif',
        "font.size": 26,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    matplotlib.rcParams.update(config)
    # matplotlib.rcParams.update({'font.size': 30})

    plt.style.use(['grid'])

    namelist = ['2006/02','2006/05','2006/08','2006/11']
    # plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(3, 2, figsize=(15, 9), sharex=True, sharey=True)

    fig.subplots_adjust(left=0.066, right=0.981, top=0.923, bottom=0.08, wspace=0.1, hspace=0.15)
    ax[0, 0].set_title('GRACE A')
    ax[0, 0].plot(np.arange(len(rmsax)), rmsax * 1000, color='black', label='Orbit arc RMS')
    ax[0, 0].axhline(y=5.8, color='#ED4C67', linestyle='--', linewidth=3, label='Average')
    ax[0, 0].legend(loc='upper right', frameon=False)
    # ax[0, 0].arrow(650, 19, -25, -11, linewidth=2, color='#ED4C67', linestyle='-', head_width=2, head_length=2)
    ax[0, 0].text(220, 17, '5.8 mm', c='#ED4C67')
    ax[1, 0].plot(np.arange(len(rmsay)), rmsay * 1000, color='black')
    ax[1, 0].axhline(y=5.4, color='#ED4C67', linestyle='--', linewidth=3)
    ax[1, 0].text(220, 17, '5.4 mm', c='#ED4C67')
    ax[2, 0].plot(np.arange(len(rmsaz)), rmsaz * 1000, color='black')
    ax[2, 0].axhline(y=5.3, color='#ED4C67', linestyle='--', linewidth=3)
    ax[2, 0].text(220, 17, '5.3 mm', c='#ED4C67')
    ax[0, 0].set(xlim=[np.arange(len(rmsax))[0], np.arange(len(rmsax))[-1]], ylim=[0, 30], ylabel='X [mm]')
    ax[1, 0].set(xlim=[np.arange(len(rmsax))[0], np.arange(len(rmsax))[-1]], ylim=[0, 30], ylabel='Y [mm]')
    ax[2, 0].set(xlim=[np.arange(len(rmsax))[0], np.arange(len(rmsax))[-1]], ylim=[0, 30], ylabel='Z [mm]')
    ax[2, 0].set_xticks([150, 500, 850, 1200])
    ax[2, 0].set_xticklabels(namelist)
    ax[0, 1].set_title('GRACE B')
    ax[0, 1].plot(np.arange(len(rmsbx)), rmsbx * 1000, color='black')
    ax[0, 1].axhline(y=5.7, color='#ED4C67', linestyle='--', linewidth=3)
    ax[0, 1].text(220, 17, '5.7 mm', c='#ED4C67')
    ax[1, 1].plot(np.arange(len(rmsby)), rmsby * 1000, color='black')
    ax[1, 1].axhline(y=5.6, color='#ED4C67', linestyle='--', linewidth=3)
    ax[1, 1].text(220, 17, '5.6 mm', c='#ED4C67')
    ax[2, 1].plot(np.arange(len(rmsbz)), rmsbz * 1000, color='black')
    ax[2, 1].axhline(y=5.5, color='#ED4C67', linestyle='--', linewidth=3)
    ax[2, 1].text(220, 17, '5.5 mm', c='#ED4C67')
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
                # print(__rms(postfit))
        # print(date_span[i][0])
        resulth5.create_dataset('rms', data=post_RMS)
        resulth5.close()
    pass


def drawkbrr():
    date_span = get_month_start_and_end_by_date_interval('2006-01-01', '2010-12-31')
    filepath = '../result/kbrr/'
    rms = []
    for i in range(len(date_span)):
        filename1 = filepath + str(date_span[i][0]) + '_' + str(date_span[i][1]) + '.hdf5'
        h5data1 = h5py.File(filename1, 'r')
        rms = np.append(rms, h5data1['rms'][:])
    # rms2 = []
    # for i in range(len(date_span)):
    #     filename2 = filepath + str(date_span[i][0]) + '_' +str(date_span[i][1]) + '-CRA.hdf5'
    #     h5data2 = h5py.File(filename2, 'r')
    #     rms2 = np.append(rms2, h5data2['rms'][:])

    config = {
        "font.family": 'serif',
        "font.size": 26,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    matplotlib.rcParams.update(config)
    # matplotlib.rcParams.update({'font.size': 30})

    plt.style.use(['grid'])
    border = plt.gca()
    border.spines['top'].set_linewidth(2)  # 设置顶部边框线宽度为2
    border.spines['right'].set_linewidth(2)  # 设置右侧边框线宽度为2
    border.spines['bottom'].set_linewidth(2)  # 设置底部边框线宽度为2
    border.spines['left'].set_linewidth(2)  # 设置左侧边框线宽度为2

    namelist = ['2006/12','2007/12','2008/12', '2009/12', '2010/12']
    plt.plot(np.arange(len(rms)), rms * 1e6, linestyle='-', marker='.', markersize=10, color='black', label='KBRR arc RMS')
    # plt.plot(np.arange(len(rms1)), rms1 * 1e6, marker='o', markersize=5, color='black', label='AOD KBRR arc RMS')
    # plt.plot(np.arange(len(rms2)), rms2 * 1e6, marker='o', markersize=5, color='red', label='CRA KBRR arc RMS')
    plt.xlim([np.arange(len(rms))[0], np.arange(len(rms))[-1]])
    plt.ylim([0.1, 0.5])
    plt.ylabel('Residual [$\mathrm{\mu m/s}$]')
    plt.xticks([1354, 2708, 4062, 5416, 6770], namelist)
    plt.legend(loc='lower right', frameon=True)
    # plt.set(xlim=, ylim=[0, 30], ylabel='X [mm]')
    plt.show()
    pass


def xyz2lonlat():
    adjust_config = AdjustOrbitConfig()
    adjustOrbitDict = json.load(open('../setting/Calibrate/AdjustOrbitConfig.json', 'r'))
    adjust_config.__dict__ = adjustOrbitDict
    interface_config = InterfaceConfig()
    interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict

    InterfacePathConfig = interface_config.PathOfFiles(interface_config)
    InterfacePathConfig.__dict__.update(interface_config.PathOfFilesConfig.copy())

    AdjustPath = AdjustOrbitConfig.PathOfFiles()
    AdjustPath.__dict__.update(adjust_config.PathOfFilesConfig.copy())

    frame_config = FrameConfig()
    frame = json.load(open('../setting/Calibrate/FrameConfig.json', 'r'))
    frame_config.__dict__ = frame

    eop = EOP().configure(frameConfig=frame_config).load()
    fr = Frame(eop).configure(frameConfig=frame_config)

    resultpath = '../result/kbrr/spatial/'
    date_span = get_month_start_and_end_by_date_interval('2009-05-01', '2009-05-31')

    for i in range(len(date_span)):
        Lon = []
        Lat = []
        Postfit = []
        raw_data_path = InterfacePathConfig.temp_raw_data
        filename = raw_data_path + '/' + date_span[i][0] + '_' + date_span[i][1] + ".hdf5"
        h5data = h5py.File(filename, "r")

        arcft_path = '../report/arcft/' + date_span[i][0] + '_' + date_span[i][1] + '.txt'
        arclist = getArcList(path=arcft_path)
        for j in arclist:
            GNV_A = h5data['GNV_A_arc'+str(j)][()]
            GNV_B = h5data['GNV_B_arc'+str(j)][()]

            GNVTime = GNV_A[:, 0]
            GNVR = GNV_A[:, 1:4]

            res_dir = pathlib.Path("../temp/RangeRate").joinpath(date_span[i][0] + '_' + date_span[i][1])
            res_filename = res_dir.joinpath('RangeRate_' + str(j) + '.hdf5')
            h5 = h5py.File(res_filename, 'r')
            t2 = h5['pre_t'][()]

            commonTime = set(GNVTime.astype(np.int64))
            commonTime = commonTime & set(t2.astype(np.int64))
            commonTime = np.array(list(commonTime))
            commonTime.sort()

            new_index1 = [list(GNVTime).index(x) for x in commonTime]
            new_index2 = [list(t2).index(x) for x in commonTime]
            new_Time = GNVTime[new_index1]
            new_GNVR = GNVR[new_index1]
            Postfit.extend(h5['pre_residual'][()][new_index2])

            for k in range(len(new_Time)):
                time = new_Time[k]
                pos = new_GNVR[k]
                fr = fr.setTime(float(time))
                itrs = fr.PosGCRS2ITRS(np.array(pos).astype(float), fr.getRotationMatrix)
                lon, lat, r = GeoMathKit.CalcPolarAngles(itrs)
                Lon.append(lon)
                Lat.append(lat)
            h5.close()
        Lon = np.degrees(np.array(Lon))
        Lat = np.degrees(np.array(Lat))
        Postfit = np.array(Postfit)

        filename = '../result/' + date_span[i][0] + '_' + date_span[i][1] + "-pre.hdf5"
        h5data = h5py.File(filename, "w")
        h5data.create_dataset('lon', data=Lon)
        h5data.create_dataset('lat', data=Lat)
        h5data.create_dataset('postfit', data=Postfit)
        h5data.close()
        h5data.close()

    pass


def getArcList(path):
    arclist = []
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(' ')
            if data[0] == 'Arc':
                if data[3] == 'True':
                    arclist.append(int(data[2]))
    return arclist


if __name__ == '__main__':
    # orbitseries()
    # draworbit()
    # kbrrseries()
    # drawkbrr()
    xyz2lonlat()