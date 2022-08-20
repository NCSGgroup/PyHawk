from src.ForceModel.BaseSecondDerivative import BaseSecondDerivative
import src.Preference.Pre_ForceModel as Pre_ForceModel
import numpy as np
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit


class AbsRelativity(BaseSecondDerivative):
    def __init__(self):
        super(AbsRelativity, self).__init__()

    def setSatState(self, pos: np.ndarray, vel: np.ndarray):
        pass

    def setConstant(self, GM_Eatrh, c_light, GM_Sun):
        pass

    def setSunState(self, pos: np.ndarray, vel: np.ndarray):
        pass

    def getAcceleration(self):
        pass

    def getSchwarzChild(self):
        pass

    def getLenseThirring(self):
        pass

    def getDesitter(self):
        pass

