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
        diffA = GNVAR - secondR[:, 0:3]
        diffB = GNVBR - secondR[:, 3:6]
        rmsA = rms(np.sqrt(diffA[:, 0] ** 2 + diffA[:, 1] ** 2 + diffA[:, 2] ** 2))
        rmsB = rms(np.sqrt(diffB[:, 0] ** 2 + diffB[:, 1] ** 2 + diffB[:, 2] ** 2))
        RMSA.append(rmsA)
        RMSB.append(rmsB)
        print(i)
        print(rmsA)
        print(rmsB)
    np.savez('2008-1-' + str(start) + '-' + str(end) + '.npz', RMSA=RMSA, RMSB=RMSB)


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
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


def draw1():
    R = np.load("2008-1RMS.npz")
    RMSA_c = R['RMSA'][0:108]
    RMSB_c = R['RMSB'][0:108]

    R1 = np.load("2008-1-0-18.npz")
    R2 = np.load("2008-1-18-36.npz")
    R3 = np.load("2008-1-36-54.npz")
    R4 = np.load("2008-1-54-72.npz")
    R5 = np.load("2008-1-72-90.npz")
    R6 = np.load("2008-1-90-108.npz")

    RMSA1 = R1['RMSA']
    RMSB1 = R1['RMSB']
    RMSA2 = R2['RMSA']
    RMSB2 = R2['RMSB']
    RMSA3 = R3['RMSA']
    RMSB3 = R3['RMSB']
    RMSA4 = R4['RMSA']
    RMSB4 = R4['RMSB']
    RMSA5 = R5['RMSA']
    RMSB5 = R5['RMSB']
    RMSA6 = R6['RMSA']
    RMSB6 = R6['RMSB']

    RMSA = np.append(RMSA1, RMSA2)
    RMSA = np.append(RMSA, RMSA3)
    RMSA = np.append(RMSA, RMSA4)
    RMSA = np.append(RMSA, RMSA5)
    RMSA = np.append(RMSA, RMSA6)

    RMSB = np.append(RMSB1, RMSB2)
    RMSB = np.append(RMSB, RMSB3)
    RMSB = np.append(RMSB, RMSB4)
    RMSB = np.append(RMSB, RMSB5)
    RMSB = np.append(RMSB, RMSB6)

    plt.title('2008-1RMS-parallel', fontsize=24)
    plt.plot(np.arange(len(RMSA)), RMSA, label="RMSA")
    plt.plot(np.arange(len(RMSA)), RMSB, label="RMSB")
    plt.xlabel('arcNo', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    plt.grid(ls='--')
    plt.show()


if __name__ == '__main__':
    # p1 = multiprocessing.Process(target=demo1, args=(0, 18))
    # p2 = multiprocessing.Process(target=demo1, args=(18, 36))
    # p3 = multiprocessing.Process(target=demo1, args=(36, 54))
    # p4 = multiprocessing.Process(target=demo1, args=(54, 72))
    # p5 = multiprocessing.Process(target=demo1, args=(72, 90))
    # p6 = multiprocessing.Process(target=demo1, args=(90, 108))
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()
    # demo1(0, 27)
    # draw()
    draw1()

