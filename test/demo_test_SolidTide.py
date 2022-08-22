from src.ForceModel.SolidTide import SolidTideIERS2010
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat
import numpy as np


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(51544.5, TimeFormat.GPS_second)
    posSun = np.array([6378136.3 * 20000, 2378136.3 * 20000, 1378136.3 * 20000])
    posMoon = np.array([637836.3 * 600, 237836.3 * 600, 137836.3 * 600])
    c, s = SolidTideIERS2010().setTime(fr).getCS(posSun, posMoon)
    print(c)
    print(s)


if __name__ == '__main__':
    demo1()