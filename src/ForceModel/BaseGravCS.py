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

    def getCS(self, **kwargs):
        return self._s, self._c

    def setTime(self, time: Frame):
        self._time = time
        return self

