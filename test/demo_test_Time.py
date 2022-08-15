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
    print(tf.getTDB_mjd)

    tf = fr.setTime(252417548.816000, TimeFormat.GPS_second)
    print('Cost time: %s ms' % ((Ti.time() - begin) * 1000))

    print(tf.getGPS_mjd)
    print(tf.getUTC_mjd)

    fr.setTime(tf.getUTC_mjd, TimeFormat.UTC_MJD)
    print(tf.getGPS_mjd)
    print(tf.getUTC_mjd)
    return True


if __name__ == '__main__':
    demo()