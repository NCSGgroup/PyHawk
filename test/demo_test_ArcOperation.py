import sys
sys.path.append('../')
from src.Interface.KinematicOrbit import KinematicOrbit
from src.Interface.L1b import GRACE_OLD_RL02, GRACE_FO_RL04, GRACE_NEW_RL02
from src.Interface.ArcOperation import ArcSelect, ArcData
from src.Frame.Frame import Frame, EOP
from src.Interface.GapFix import GapFix
from src.Preference.EnumType import Payload, SatID
import src.Preference.EnumType as EnumType


def demo1():

    fr = Frame(EOP().load())
    s = GRACE_OLD_RL02().configure().setDate()
    s.read_double_sat()

    kine = KinematicOrbit(L1b=s).configure()
    kine.read_double_sat()
    kine.GNVandKiOrbitItrs2Gcrs(fr)

    gf = GapFix(s).configure()
    gf.fix_all().makeReport()

    #
    arc = ArcSelect(gf).configure()
    arc.unpackArcs().makeReport()

    pass


def demo2():
    rd = ArcData()
    res = rd.getData(arc=0, kind=Payload.GNV, sat=SatID.A)
    print(res)
    pass


if __name__ == '__main__':
    demo1()
    # demo2()
