import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.ForceModel.AtmosTide import AbsAtmosTide, AtmosBB2003, AtmosAOD
from src.ForceModel.NonTide import AODRL06
from src.ForceModel.OceanTide import AbsOceanTide, EOT11a, FES2014
from src.ForceModel.PoleTide import AbsPoleTide, PoleTideIERS2010
from src.ForceModel.SolidTide import SolidTideIERS2010
from src.ForceModel.RefGravModel import RefGravModel
from src.Preference.EnumType import ForceType, GravHarType
from src.ForceModel.BaseGravCS import BaseGravCS
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit
from src.ForceModel.PlanetEphemerides import PlanetEphemerides
from src.ForceModel.GravHar import ClibGravHar
from src.Preference.EnumType import InstanceNonTide, InstanceAtmosTide, InstanceOceanTide, \
    InstancePoleTide, InstanceSolidTide, InstanceReferenceModel
from src.Interface.LoadSH import LoadGif48


class ConfigGravHar:

    def __init__(self):
        self.__include_GravHarType = None
        self.__isFirst = True
        self.__sTotal = None
        self.__cTotal = None
        self.__time = None
        self.__ep = None
        self.__refmodel = None
        self.GravHarForce = {}

        include_GravHarType = Pre_ForceModel.include_GravHarType
        instance_GravHar = Pre_ForceModel.instance_GravHar
        self.__include_GravHarType = [key for key, value in include_GravHarType.items() if value]
        self.__instance_GravHar = instance_GravHar
        pass

    def setTime(self, time: Frame, ep: PlanetEphemerides):
        self.__time = time
        self.__ep = ep
        self.__refmodel = LoadGif48().load()
        self.__instanceForce()
        for obj in self.GravHarForce.values():
            self.accumulate(obj)
        return self

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
            if tide is GravHarType.OceanTide:
                obj = None
                if self.__instance_GravHar[tide] is InstanceOceanTide.EOT11a:
                    obj = EOT11a().setTime(self.__time).load()
                elif self.__instance_GravHar[tide] is InstanceOceanTide.FES2014:
                    obj = FES2014().setTime(self.__time).load()
                self.GravHarForce[tide] = obj
            if tide is GravHarType.AtmosphereTide:
                obj = None
                if self.__instance_GravHar[tide] is InstanceAtmosTide.AtmosBB2003:
                    obj = AtmosBB2003().setTime(self.__time)
                elif self.__instance_GravHar[tide] is InstanceAtmosTide.AtmosAOD:
                    obj = AtmosAOD().setTime(self.__time).loadfile()
                self.GravHarForce[tide] = obj
            if tide is GravHarType.PoleTide:
                obj = None
                if self.__instance_GravHar[tide] is InstancePoleTide.PoleTideIERS2010:
                    obj = PoleTideIERS2010().load().setTime(self.__time)
                self.GravHarForce[tide] = obj
            if tide is GravHarType.SolidEarthTide:
                obj = None
                if self.__instance_GravHar[tide] is InstanceSolidTide.SolidTideIERS2010:
                    obj = SolidTideIERS2010(self.__ep).setTime(self.__time)
                self.GravHarForce[tide] = obj
            if tide is GravHarType.NonTide:
                obj = None
                if self.__instance_GravHar[tide] is InstanceNonTide.AODRL06:
                    obj = AODRL06().setTime(self.__time).configure()
                self.GravHarForce[tide] = obj
            if tide is GravHarType.ReferenceModel:
                obj = None
                if self.__instance_GravHar[tide] is InstanceReferenceModel.RefGravModel:
                    obj = RefGravModel().setTime(self.__time).setModel(self.__refmodel)
                self.GravHarForce[tide] = obj

