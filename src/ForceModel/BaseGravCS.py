from src.Frame.Frame import Frame


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

