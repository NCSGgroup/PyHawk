from src.Frame.EOP import EOP
from src.Frame.Frame import Frame
import src.Frame.SOFA as sf
import time as Ti
import numpy as np
from src.Preference.EnumType import TimeFormat


def demo():
    fr = Frame(EOP(True).load())
    # self.__tdbOption = 2
    begin = Ti.time()
    tf = fr.setTime(fr.cal2mjd(2007, 1, 3, 18, 0, 0), TimeFormat.UTC_MJD)

    pos_GCRS = np.array([100000., 200000., 300000.])
    vel_GCRS = np.array([100., 200., 300.])

    pos_ITRS = fr.PosGCRS2ITRS(pos_GCRS, tf.getRotationMatrix)
    # pos_GCRS = fr.PosITRS2GCRS(pos_ITRS, tf.getRotationMatrix)

    vel_ITRS = fr.VelGCRS2ITRS(pos_GCRS, vel_GCRS, tf.getRotationMatrix)
    # vel_GCRS = fr.VelITRS2GCRS(pos_ITRS, vel_ITRS, tf.getRotationMatrix)
    print(pos_ITRS)
    print(vel_ITRS)

    return True


if __name__ == '__main__':
    demo()