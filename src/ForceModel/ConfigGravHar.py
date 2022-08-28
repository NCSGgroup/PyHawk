import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.ForceModel.AtmosTide import AbsAtmosTide
from src.ForceModel.NonTide import AODRL06
from src.ForceModel.OceanTide import AbsOceanTide
from src.ForceModel.PoleTide import AbsPoleTide
from src.ForceModel.SolidTide import SolidTideIERS2010
from src.Preference.EnumType import ForceType, GravHarType
from src.ForceModel.BaseGravCS import BaseGravCS
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit


class ConfigGravHar:

    def __init__(self):
        self.__include_GravHarType = None
        self.__isFirst = True
        self.__sTotal = None
        self.__cTotal = None
        self.__time = None
        self.GravHarForce = {}

        include_GravHarType = Pre_ForceModel.include_GravHarType
        instance_GravHar = Pre_ForceModel.instance_GravHar
        self.__include_GravHarType = [key for key, value in include_GravHarType.items() if value]
        self.__instance_GravHar = instance_GravHar
        pass

    def setTime(self, time: Frame):
        self.__time = time
        self.__instanceForce()

    def accumulate(self, baseCS: BaseGravCS):
        C, S = baseCS.getCS()

        if self.__isFirst:
            self.__cTotal = C
            self.__sTotal = S
            self.__isFirst = False
        else:
            self.__cTotal = GeoMathKit.arrayAdd(C, self.__cTotal)
            self.__sTotal = GeoMathKit.arrayAdd(S, self.__sTotal)

        return self

    def getSumCS(self):
        return self.__cTotal, self.__sTotal

    def __instanceForce(self):
        for tide in self.__include_GravHarType:
            self.GravHarForce[tide] = BaseGravCS.new_instance(tide, self.__instance_GravHar[tide]).setTime(self.__time)





