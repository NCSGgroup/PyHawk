import numpy as np
from src.Frame.Frame import Frame
import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.Auxilary.GeoMathKit import GeoMathKit


class BaseForceCS:

    def __init__(self):
        self._s = None
        self._c = None
        self._time = None
        self._sTotal = None
        self._cTotal = None
        self._isFirst = True

    def getCS(self):
        return self._s, self._c

    def getTotalCS(self):
        return self._cTotal, self._sTotal

    def accumulate(self):
        if self._isFirst:
            self._cTotal = self._c
            self._sTotal = self._s
            self._isFirst = False
        else:
            self._cTotal = GeoMathKit.arrayAdd(self._c, self._cTotal)
            self._sTotal = GeoMathKit.arrayAdd(self._s, self._sTotal)
        return self

    def setState(self, r, v):
        pass

    def setTime(self, time: Frame):
        self._time = time
        return self

