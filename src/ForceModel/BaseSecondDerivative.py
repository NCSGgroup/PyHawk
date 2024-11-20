import numpy as np
from src.Frame.Frame import Frame


class BaseSecondDerivative:

    def __init__(self):
        self._zeroOrder = None
        self._firstOrder = None
        self._odeVariableType = None
        self._time = None
        self._rm = None
        pass

    def setTime(self, time: Frame):
        self._time = time
        return self

    def setRotationMatrix(self, rm):
        self._rm = rm
        return self

    def changeSat(self):
        return self

    def getAcceleration(self, **kwargs):
        acc = np.zeros(3, dtype=float)
        return acc

    def getDaDx(self, **kwargs):
        dadx = np.zeros(3, dtype=float)
        return dadx

    def getDaDp(self, **kwargs):
        """
        dax/dp1  dax/dp2  ...
        day/dp1  day/dp2  ...
        daz/dp1  daz/dp2  ...
        :return:  da/dp in GCRS frame [3*np]
        """
        dadp = np.zeros(3, dtype=float)
        return dadp

