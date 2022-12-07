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


def demo1():

    # """step 1: Read raw data"""
    # fr = Frame(EOP().load())
    # s = GRACE_NEW_RL02().configure().setDate()
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
    adjust = AdjustOrbit().configure().calibrate(iteration=3)
    # initTime, initR, endTime, endR = adjust.draw()
    firstR, secondR, thirdR = adjust.getR()

    # initR= initR[:, 0:3, 0]
    # endR= endR[:, 0:3, 0]
    rd = ArcData()
    GNV = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    GNVTime = GNV[:, 0]
    GNVR = GNV[:, 1:4]

    np.savez('adjust.npz', time=GNVTime, GNVR=GNVR, R1=firstR, R2=secondR, R3=thirdR)


def draw():
    R = np.load("adjust.npz")
    Time = R['time']
    GNVR = R['GNVR']
    firstR = R['R1'][:, 0:3, 0]
    secondR = R['R2'][:, 0:3, 0]
    thirdR = R['R3'][:, 0:3, 0]

    fig = plt.figure()
    # plt.style.use(['science', 'grid', 'no-latex'])
    plt.title('nine par adjust', fontsize=24)
    # np.sqrt(np.sum((np.array(GNVR[:]) - endR[:, 0:3]) ** 2, 1))

    plt.plot(np.array(Time), np.sqrt(np.sum((np.array(GNVR[:]) - firstR[:, 0:3]) ** 2, 1)), label="first adjust")
    plt.plot(np.array(Time), np.sqrt(np.sum((np.array(GNVR[:]) - secondR[:, 0:3]) ** 2, 1)), label="second adjust")
    plt.plot(np.array(Time), np.sqrt(np.sum((np.array(GNVR[:]) - thirdR[:, 0:3]) ** 2, 1)), label="third adjust")

    plt.xlabel('GPS time [sec]', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    # plt.grid(ls='--')
    plt.show()


if __name__ == '__main__':
    # demo1()
    draw()



