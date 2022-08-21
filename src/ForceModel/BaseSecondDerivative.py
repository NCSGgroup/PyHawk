import numpy as np
from src.Frame.Frame import Frame


class BaseSecondDerivative:

    def __init__(self):
        self._zeroOrder = None
        self._firstOrder = None
        self._odeVariableType = None
        self._time = None
        pass

    def setTime(self, time: Frame):
        self._time = time
        return self

    # def getSecondDerivative(self):
    #
    #     pass

    # def setArguments(self, zeroOrder, firstOrder, odeVariableType):
    #     self._zeroOrder = zeroOrder
    #     self._firstOrder = firstOrder
    #     self._odeVariableType = odeVariableType
    #     pass

    def getAcceleration(self, **kwargs):
        acc = None
        return acc

    def getDaDxyz(self, **kwargs):
        dadx = None
        return dadx

    def getDaDp(self, **kwargs):
        """
        dax/dp1  dax/dp2  ...
        day/dp1  day/dp2  ...
        daz/dp1  daz/dp2  ...
        :return:  da/dp in GCRS frame [3*np]
        """
        dadp = None
        return dadp

