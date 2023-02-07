import os,sys
from src.Interface.ArcOperation import ArcSelect, ArcData
import src.Preference.EnumType as EnumType
import numpy as np
import time
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.Solver.AdjustOrbit import AdjustOrbit
import matplotlib.pyplot as plt
import multiprocessing
sys.path.append("../")


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
    rd = ArcData()
    RMSA = []
    RMSB = []
    for i in range(start, end):
        GNVA = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
        GNVTime = GNVA[:, 0]
        GNVAR = GNVA[:, 1:4]
        GNVB = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat=EnumType.SatID.B)
        GNVBR = GNVB[:, 1:4]

        adjust = AdjustOrbit(arcNo=i).configure(Orbit_var_2nd_diff).calibrate(iteration=2)
        # initTime, initR, endTime, endR = adjust.draw()
        firstR, secondR, thirdR = adjust.getR()
        secondR = secondR[:, 0:6, 0]
        rmsA = rms(np.array(GNVAR) - secondR[:, 0:3])
        rmsB = rms(np.array(GNVBR) - secondR[:, 3:6])
        RMSA.append(rmsA)
        RMSB.append(rmsB)
        print(i)
        print(rmsA)
        print(rmsB)
    np.savez('2008-1-' + start + '-' + end + '.npz', RMSA=RMSA, RMSB=RMSB)


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


def draw():
    R = np.load("2008-1RMS.npz")
    RMSA = R['RMSA']
    RMSB = R['RMSB']
    plt.title('2008-1RMS', fontsize=24)

    plt.plot(np.arange(len(RMSA)), RMSA, label="RMSA")
    plt.plot(np.arange(len(RMSA)), RMSB, label="RMSB")
    plt.xlabel('GPS time [sec]', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    # plt.grid(ls='--')
    plt.show()


if __name__ == '__main__':
    p1 = multiprocessing.Process(target=demo1, args=(0, 27))
    p2 = multiprocessing.Process(target=demo1, args=(27, 54))
    p3 = multiprocessing.Process(target=demo1, args=(54, 81))
    p4 = multiprocessing.Process(target=demo1, args=(81, 108))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    # demo1()
    # draw()

