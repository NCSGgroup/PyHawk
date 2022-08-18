from src.Frame.Frame import Frame
import src.Preference.Pre_ForceModel as Pre_ForceModel


class BaseForceCS:

    def __init__(self):
        self._s = None
        self._c = None
        self._fr = None
        self._sTotal = None
        self._cTotal = None

    def getCS(self):
        return self._s, self._c

    def getTotalCS(self):
        return self._cTotal, self._sTotal

    def accumulate(self):
        # self._cTotal += self._c
        # self._sTotal += self._s
        print(self._cTotal)
        print(self._c)
        return self

    def setState(self, r, v):
        pass

    def setTime(self, time: Frame):
        self._fr = time
        return self

