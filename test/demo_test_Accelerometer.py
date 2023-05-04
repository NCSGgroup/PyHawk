from src.Interface.Accelerometer import Accelerometer, AccCaliPar
from src.Interface.ArcOperation import ArcData
from src.Preference.EnumType import SatID, Payload
import numpy as np


def demo1():
    sat = SatID.B
    rd = ArcData()
    ap = AccCaliPar(sat=sat, arcNo=0)

    ac = Accelerometer(rd=rd, ap=ap)
    ac.get_and_save()
    res0 = ac.get_acceleration()
    res1 = ac.get_du2p()
    print(res0)
    pass


if __name__ == '__main__':
    demo1()
