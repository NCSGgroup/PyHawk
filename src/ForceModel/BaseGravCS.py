import numpy as np
from src.Frame.Frame import Frame
import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Preference.EnumType import GravHarType


class BaseGravCS:

    def __init__(self):
        self._s = None
        self._c = None
        self._time = None
        self._sTotal = None
        self._cTotal = None
        self._isFirst = True
        self._tideType = None

    def setType(self, tideType:GravHarType):
        self._tideType = tideType
        return self

    def getType(self):
        return self._tideType

    def getCS(self, **kwargs):
        return self._s, self._c

    def getTotalCS(self):
        return self._cTotal, self._sTotal

    def accumulate(self, c, s):
        self._s = s
        self._c = c
        if self._isFirst:
            self._cTotal = self._c
            self._sTotal = self._s
            self._isFirst = False
        else:
            self._cTotal = GeoMathKit.arrayAdd(self._c, self._cTotal)
            self._sTotal = GeoMathKit.arrayAdd(self._s, self._sTotal)
        return self

    def setTime(self, time: Frame):
        self._time = time
        return self

