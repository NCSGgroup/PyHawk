import numpy as np

from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.Frame.Frame import Frame
from src.Preference.EnumType import TimeFormat


class Orbit_2nd_diff(Assemble2ndDerivative):

    def __init__(self, fr: Frame):
        self.__fr = fr
        super(Orbit_2nd_diff, self).__init__()

    def secDerivative(self, t, r, v):
        """
        For orbit only in terms of 2nd order differential equation
        :param v: vel vector
        :param t: time
        :param r: position vector
        :return:
        """

        self.__fr = self.__fr.setTime(t, TimeFormat.GPS_second)
        self.setPosAndVel(r[0], v[0]).setTime(self.__fr)
        accA = self.getAcceleration()

        self.setPosAndVel(r[1], v[1]).setTime(self.__fr)
        accB = self.getAcceleration()

        return np.vstack((accA, accB))




