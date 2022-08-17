from src.Frame.Frame import Frame
import src.Preference.Pre_ForceModel as Pre_ForceModel


class BaseForceCS:
    def __init__(self):
        self._csTotal = None
        self._cs = None

    def getCS(self):
        return self

    def accumulate(self):
        self._csTotal += self._cs

    def setState(self, r, v):
        pass

    def setTime(self, time: Frame):
        pass

