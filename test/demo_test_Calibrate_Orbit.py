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
    adjust = AdjustOrbit().configure().calibrate()


def demo2():
    pass


if __name__ == '__main__':
    demo1()
    # demo2()

