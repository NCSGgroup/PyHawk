import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.ForceModel.BaseGravCS import BaseGravCS
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit


class ConfigGravHar:

    def __init__(self):
        self.include_ForceType = None
        self._isFirst = True
        self._sTotal = None
        self._cTotal = None
        pass

    def configure(self, **kwargs):
        include_GravHarType = kwargs.get('include_ForceType', Pre_ForceModel.include_GravHarType)
        self.include_ForceType = [key for key, value in include_GravHarType.items() if value]
        return self

    def accumulate(self, baseCS: BaseGravCS):
        C,S = None, None
        if baseCS.getType() in self.include_ForceType:
            C, S = baseCS.getCS()

        if self._isFirst:
            self._cTotal = C
            self._sTotal = S
            self._isFirst = False
        else:
            self._cTotal = GeoMathKit.arrayAdd(C, self._cTotal)
            self._sTotal = GeoMathKit.arrayAdd(S, self._sTotal)

        return self

    def getSumCS(self):
        return self._cTotal, self._sTotal


