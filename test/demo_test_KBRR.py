import sys
sys.path.append("../")
from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType
import numpy as np
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.Solver.AdjustOrbit import AdjustOrbit
import pathlib
import h5py
import multiprocessing
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Preference.Pre_Accelerometer import AccelerometerConfig
from src.Preference.Pre_Solver import SolverConfig
from src.Preference.Pre_Parameterization import ParameterConfig
import json
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Interface.L1b import GRACE_OLD_RL02
from src.Interface.KinematicOrbit import KinematicOrbitV3
from src.Interface.GapFix import GapFix
from src.Interface.ArcOperation import ArcSelect
from src.Solver.AdjustSST import RangeRate
from src.Interface.ArcOperation import ArcData
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_Frame import FrameConfig
import os
from datetime import datetime
import datetime
import calendar
import pygmt


def AODRL06():
    """step 1: Read raw data"""
    date_span = ['2008-01-01', '2008-01-02']
    date_list = GeoMathKit.getEveryDay(*date_span)
    days = datetime.strptime(date_list[-1], '%Y-%m-%d') - datetime.strptime(date_list[0], '%Y-%m-%d')
    arcLen = days.days * 8
    kwargs = {'date_span': date_span}
    fr = Frame(EOP().load())
    s = GRACE_OLD_RL02().configure().setDate(**kwargs)
    s.read_double_sat()
    """step 2: Convert orbit from ITRS frame into GCRS frame"""
    kine = KinematicOrbitV3(L1b=s).configure()
    # kine.read_double_sat()
    kine.GNVandKiOrbitItrs2Gcrs(fr)
    """step 3: Gap fix"""
    gf = GapFix(s).configure()
    gf.fix_all().makeReport()
    arc = ArcSelect(gf).configure()
    arc.unpackArcs().makeReport()
    """step 4: config"""
    accConfig_A = AccelerometerConfig()
    acc_A = json.load(open('../setting/Calibrate/AccelerometerConfig_A.json', 'r'))
    accConfig_A.__dict__ = acc_A
    accConfig_B = AccelerometerConfig()
    acc_B = json.load(open('../setting/Calibrate/AccelerometerConfig_B.json', 'r'))
    accConfig_B.__dict__ = acc_B
    solverConfig = SolverConfig()
    solverDict = json.load(open('../setting/Calibrate/SolverConfig.json', 'r'))
    solverConfig.__dict__ = solverDict
    parameterConfig = ParameterConfig()
    Parameter = json.load(open('../setting/Calibrate/ParameterConfig.json', 'r'))
    parameterConfig.__dict__ = Parameter
    """step 5: init AdjustOrbit"""
    for i in range(0, arcLen):
        AdjustOrbit(accConfig=[accConfig_A, accConfig_B],date_span=date_span).\
            configure(SerDer=Orbit_var_2nd_diff, arcNo=i, ParameterConfig=parameterConfig).calibrate(iteration=2)
    """step 6: adjustSST"""
    for i in range(0, arcLen):
        RR = RangeRate(i, date_span=date_span)
        RR.configure()
        RR.calibrate()
    """step 7: drawKBRR"""
    drawKBRR(datespan=date_span, arcLen=arcLen)


def drawKBRR(datespan, arcLen):
    fr = Frame(EOP().load())
    rd = ArcData(date_span=datespan)
    Lon = []
    Lat = []
    Postfit = []
    for i in range(0, arcLen):
        GNV = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
        GNVTime = GNV[:, 0]
        GNVR = GNV[:, 1:4]

        res_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.RangeRateTemp).joinpath(datespan[0] + '_' + datespan[1])
        res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        t2 = h5['post_t'][()]
        Postfit.extend(h5['post_residual'][()])

        new_index = [list(GNVTime).index(x) for x in t2]
        new_Time = GNVTime[new_index]
        new_GNVR = GNVR[new_index]
        # new_KBRR = h5['pre_residual'][()][new_index]
        for i in range(len(new_Time)):
            time = new_Time[i]
            pos = new_GNVR[i]
            fr = fr.setTime(float(time))
            itrs = fr.PosGCRS2ITRS(np.array(pos).astype(float), fr.getRotationMatrix)
            lon, lat, r = GeoMathKit.CalcPolarAngles(itrs)
            Lon.append(lon)
            Lat.append(lat)

    Lon = np.degrees(np.array(Lon))
    Lat = np.degrees(np.array(Lat))
    Postfit = np.array(Postfit) * 1e7
    grid = pygmt.xyz2grd(x=Lon, y=Lat, z=Postfit, spacing=(1, 1), region="d", duplicate='r')
    fig = pygmt.Figure()
    pygmt.makecpt(series=[0, 7, 1])
    fig.grdimage(grid=grid, region="d", projection="Q12c", frame="afg")
    fig.coast(shorelines="1/0.5p", projection="Q12c")
    fig.colorbar()
    fig.savefig('../result/kbrr/KBRR-space distribution.jpg')
    fig.show()


def xyz2lonlat():
    datespan = get_month_start_and_end_by_date_interval('2009-05-01', '2009-05-31')

    for i in range(len(datespan)):
        frame_config = FrameConfig()
        frame = json.load(open('../setting/Calibrate/FrameConfig.json', 'r'))
        frame_config.__dict__ = frame

        eop = EOP().configure(frameConfig=frame_config).load()
        fr = Frame(eop).configure(frameConfig=frame_config)

        raw_data_path = '../temp/RawData'
        filename = pathlib.Path().joinpath(raw_data_path, datespan[i][0] + '_' + datespan[i][1] + ".hdf5")
        h5data = h5py.File(filename, "r")

        Lon = []
        Lat = []
        Postfit = []
        path = pathlib.Path().joinpath('../report/arcft', datespan[i][0] + '_' + datespan[i][1]+'.txt')
        arclen = getArcList(path=path)
        for j in arclen:
            GNV = h5data['GNV_A_arc' + str(j)][()]
            GNVTime = GNV[:, 0]
            GNVR = GNV[:, 1:4]

            res_dir = pathlib.Path("D:/py-hawk-v2/temp/RangeRate/acc-2h/").joinpath(datespan[i][0] + '_' + datespan[i][1])
            res_filename = res_dir.joinpath('RangeRate_' + str(j) + '.hdf5')
            if not os.path.exists(res_filename):
                continue
            h5 = h5py.File(res_filename, 'r')
            t2 = h5['post_t'][()]

            commonTime = set(GNVTime.astype(np.int64))
            commonTime = commonTime & set(t2.astype(np.int64))
            commonTime = np.array(list(commonTime))
            commonTime.sort()

            new_index1 = [list(GNVTime).index(x) for x in commonTime]
            new_index2 = [list(t2).index(x) for x in commonTime]
            new_Time = GNVTime[new_index1]
            new_GNVR = GNVR[new_index1]
            Postfit.extend(h5['post_residual'][()][new_index2])

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

        filename = 'D:/py-hawk-v2/temp/RangeRate/acc-2h/' + datespan[i][0] + '_' + datespan[i][1] + '.hdf5'
        h5data = h5py.File(filename, "w")
        h5data.create_dataset('lon', data=Lon)
        h5data.create_dataset('lat', data=Lat)
        h5data.create_dataset('postfit', data=Postfit)
        h5data.close()
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
    # AODRL06()
    xyz2lonlat()
    # filename = '../result/2008-07-CRA.hdf5'
    # data = h5py.File(filename, "r")
    # data.keys()
