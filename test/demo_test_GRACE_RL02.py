from src.Frame.Frame import Frame, EOP
from src.Interface.L1b import GRACE_OLD_RL02, GRACE_NEW_RL02
import src.Preference.EnumType as EnumType
import numpy as np
from src.Auxilary.GeoMathKit import GeoMathKit


def demo1():
    fr = Frame(EOP().load())
    s = GRACE_NEW_RL02().configure().setDate()
    # print(s.getDate())
    s.read_double_sat()

    GNVA = s.getData(EnumType.Payload.GNV, EnumType.SatID.A)

    print(np.shape(GNVA))
    for xyz in GNVA[:, 1:4]:
        print(GeoMathKit.CalcPolarAngles(xyz))

    # s.getData()
    # s.clear()
    pass


if __name__ == '__main__':
    demo1()