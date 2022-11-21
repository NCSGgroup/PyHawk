from src.Interface.KinematicOrbit import KinematicOrbit
from src.Interface.L1b import GRACE_OLD_RL02, GRACE_FO_RL04, GRACE_NEW_RL02
from src.Interface.ArcOperation import ArcSelect, ArcData
from src.Frame.Frame import Frame, EOP
from src.Interface.GapFix import GapFix
import src.Preference.EnumType as EnumType
import numpy as np
import time
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.ODE.Instance.GravimetryODE import GravimetryODE
from src.Interface.Accelerometer import Accelerometer, AccCaliPar


def demo1():

    """step 1: Read raw data"""
    fr = Frame(EOP().load())
    s = GRACE_NEW_RL02().configure().setDate()
    s.read_double_sat()
    """step 2: Convert orbit from ITRS frame into GCRS frame"""
    kine = KinematicOrbit(L1b=s).configure()
    kine.read_double_sat()
    kine.GNVandKiOrbitItrs2Gcrs(fr)
    """step 3: Gap fix"""
    gf = GapFix(s).configure()
    gf.fix_all().makeReport()
    arc = ArcSelect(gf).configure()
    arc.unpackArcs().makeReport()
    """step 4: combine the SCA and ACC data to obtain the non-conservative force as well as dadp arc by arc"""
    rd = ArcData()
    ap = AccCaliPar(sat=EnumType.SatID.A, arcNo=0)
    ac = Accelerometer(rd=rd, ap=ap)
    ac.get_and_save()
    """step 5: obtain the GNV orbit of GRACE to initialize the ODE"""
    resA = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    resB = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.B)
    GPS_Time = resA[4, 0]
    startPosA = resA[4, 1:4]
    startVelA = resA[4, 4:7]
    startPosB = resB[4, 1:4]
    startVelB = resB[4, 4:7]
    startPos = np.vstack((startPosA, startPosB))
    startVel = np.vstack((startVelA, startVelB))
    """step 6: instance second Derivative"""
    var_2nd_diff = Orbit_var_2nd_diff(fr)
    """step 7: Propagator"""
    time1 = time.time()
    propagator = GravimetryODE().setInitial(GPS_Time, startPos, startVel).set2ndDerivative(var_2nd_diff)
    # X_GJ, R_GJ, V_GJ = propagator.propagate()


if __name__ == '__main__':
    demo1()

