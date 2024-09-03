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
from datetime import datetime
from src.Interface.ArcOperation import ArcData
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_Frame import FrameConfig

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
    datespan = ["2020-04-01", "2020-04-30"]
    interface_config = InterfaceConfig()
    interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict

    frame_config = FrameConfig()
    frame = json.load(open('../setting/Calibrate/FrameConfig.json', 'r'))
    frame_config.__dict__ = frame

    eop = EOP().configure(frameConfig=frame_config).load()
    fr = Frame(eop).configure(frameConfig=frame_config)

    rd = ArcData(interfaceConfig=interface_config)
    Lon = []
    Lat = []
    Postfit = []
    for i in range(0, 116):
        GNV = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
        GNVTime = GNV[:, 0]
        GNVR = GNV[:, 1:4]

        res_dir = pathlib.Path("../temp/RangeRate/2020-4/Normal").joinpath(datespan[0] + '_' + datespan[1])
        res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
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

        for i in range(len(new_Time)):
            time = new_Time[i]
            pos = new_GNVR[i]
            fr = fr.setTime(float(time))
            itrs = fr.PosGCRS2ITRS(np.array(pos).astype(float), fr.getRotationMatrix)
            lon, lat, r = GeoMathKit.CalcPolarAngles(itrs)
            Lon.append(lon)
            Lat.append(lat)
        h5.close()

    Lon = np.degrees(np.array(Lon))
    Lat = np.degrees(np.array(Lat))
    Postfit = np.array(Postfit)

    filename = '../result/2020-04-Normal.hdf5'
    h5data = h5py.File(filename, "w")
    h5data.create_dataset('lon', data=Lon)
    h5data.create_dataset('lat', data=Lat)
    h5data.create_dataset('postfit', data=Postfit)
    h5data.close()
    pass


if __name__ == '__main__':
    # AODRL06()
    xyz2lonlat()
    # filename = '../result/2008-07-CRA.hdf5'
    # data = h5py.File(filename, "r")
    # data.keys()
