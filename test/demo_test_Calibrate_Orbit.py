import os,sys
sys.path.append("../")
from src.Interface.KinematicOrbit import KinematicOrbit
from src.Interface.L1b import GRACE_OLD_RL02, GRACE_FO_RL04, GRACE_NEW_RL02
from src.Interface.ArcOperation import ArcSelect, ArcData
from src.Frame.Frame import Frame, EOP
from src.Interface.GapFix import GapFix
import src.Preference.EnumType as EnumType
import numpy as np
import time
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.Solver.AdjustOrbit import AdjustOrbit
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.ODE.Instance.GravimetryODE import GravimetryODE
from src.Interface.Accelerometer import Accelerometer, AccCaliPar
import src.Preference.Pre_Interface as Pre_Interface
import matplotlib.pyplot as plt
import src.Preference.Pre_AdjustOrbit as Pre_AdjustOrbit
import pathlib
import h5py
import multiprocessing


def demo1(start, end):

    # """step 1: Read raw data"""
    # fr = Frame(EOP().load())
    # s = GRACE_OLD_RL02().configure().setDate()
    # s.read_double_sat()
    # """step 2: Convert orbit from ITRS frame into GCRS frame"""
    # kine = KinematicOrbit(L1b=s).configure()
    # kine.read_double_sat()
    # kine.GNVandKiOrbitItrs2Gcrs(fr)
    # """step 3: Gap fix"""
    # gf = GapFix(s).configure()
    # gf.fix_all().makeReport()
    # arc = ArcSelect(gf).configure()
    # arc.unpackArcs().makeReport()
    """step 5: init AdjustOrbit"""
    for i in range(start,end):
        adjust = AdjustOrbit(arcNo=i).configure(Orbit_var_2nd_diff).calibrate(iteration=2)
        # initTime, initR, endTime, endR = adjust.draw()
        # firstR, secondR, thirdR = adjust.getR()
        # np.savez('adjust.npz', R1=firstR, R2=secondR, R3=thirdR)


def draw():
    dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp). \
        joinpath('2008-01-01_2008-01-31')
    filename = dir.joinpath(str(100) + '_AB.hdf5')
    h5 = h5py.File(filename, 'r')
    orbit_nominal = h5['r'][()][:,3:6,0]
    print(np.shape(orbit_nominal))
    h5.close()

    rd = ArcData()
    GNV = rd.getData(arc=100, kind=EnumType.Payload.GNV, sat=EnumType.SatID.B)
    GNVTime = GNV[:, 0]
    GNVR = GNV[:, 1:4]

    plt.title('2008-1-1 sat-A arc-0 (constant and trend)', fontsize=24)
    plt.plot(np.array(GNVTime), np.array(GNVR[:, 0]) - orbit_nominal[:, 0], label="X")
    plt.plot(np.array(GNVTime), np.array(GNVR[:, 1]) - orbit_nominal[:, 1], label="Y")
    plt.plot(np.array(GNVTime), np.array(GNVR[:, 2]) - orbit_nominal[:, 2], label="Z")
    plt.xlabel('GPS time [sec]', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    # plt.grid(ls='--')
    plt.show()


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


if __name__ == '__main__':
    # demo1()
    # draw()
    p1 = multiprocessing.Process(target=demo1, args=(0, 18))
    p2 = multiprocessing.Process(target=demo1, args=(18, 36))
    p3 = multiprocessing.Process(target=demo1, args=(36, 54))
    p4 = multiprocessing.Process(target=demo1, args=(54, 72))
    p5 = multiprocessing.Process(target=demo1, args=(72, 90))
    p6 = multiprocessing.Process(target=demo1, args=(90, 108))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()

