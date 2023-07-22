from src.Interface.KinematicOrbit import KinematicOrbitV3
from src.Interface.L1b import GRACE_OLD_RL02, GRACE_FO_RL04
from src.Frame.Frame import Frame, EOP
from src.Interface.GapFix import GapFix


def demo1():
    fr = Frame(EOP().load())

    s = GRACE_FO_RL04().configure().setDate()
    s.read_double_sat()

    kine = KinematicOrbitV3(L1b=s).configure()
    kine.read_double_sat()
    kine.GNVandKiOrbitItrs2Gcrs(fr)

    gf = GapFix(s).configure()
    gf.fix_all().makeReport()
    pass


if __name__ == '__main__':
    demo1()
