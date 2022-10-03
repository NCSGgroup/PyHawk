import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.ForceModel.AtmosTide import AtmosBB2003, AtmosAOD
from src.ForceModel.NonTide import AODRL06
from src.ForceModel.OceanTide import EOT11a, FES2014
from src.ForceModel.PoleTide import PoleTideIERS2010
from src.ForceModel.SolidTide import SolidTideIERS2010
from src.ForceModel.RefGravModel import RefGravModel, ConfigStaticGravModel
from src.Preference.EnumType import GravHarType
from src.ForceModel.BaseGravCS import BaseGravCS
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Frame.PlanetEphemerides import PlanetEphemerides
import src.Preference.EnumType as EnumType
from src.ForceModel.GravHar import ClibGravHar


class ConfigGravHar:

    def __init__(self, ep: PlanetEphemerides):
        self.__include_GravHarType = None
        self.__isFirst = True
        self.__sTotal = None
        self.__cTotal = None
        self.__time = None
        self.__ep = ep
        self.setZero = Pre_ForceModel.GravHarConfig.setZero.copy()
        self.InputPosCoordinate = Pre_ForceModel.GravHarConfig.InputPosCoordinate
        self.OutputCoordinate = Pre_ForceModel.GravHarConfig.OutputCoordinate
        self.GravHarForce = {}
        self.ClibGravHar = ClibGravHar().setConstant()

        include_GravHarType = Pre_ForceModel.include_GravHarType.copy()
        instance_GravHar = Pre_ForceModel.instance_GravHar.copy()
        self.__include_GravHarType = [key for key, value in include_GravHarType.items() if value]
        self.__instance_GravHar = instance_GravHar

        self.__instance_force()
        pass

    def __reset(self):
        self.__isFirst = True
        self.__sTotal = None
        self.__cTotal = None

    def setTime(self, time: Frame):
        self.__reset()
        self.__time = time
        for obj in self.GravHarForce.values():
            obj = obj.setTime(time)
            self.__accumulate(obj)
        return self

    def getTotalCS(self):
        return self.__cTotal, self.__sTotal

    def __accumulate(self, baseCS: BaseGravCS):
        C, S = baseCS.getCS()
        if self.__isFirst:
            self.__cTotal = C
            self.__sTotal = S
            self.__isFirst = False
        else:
            self.__cTotal = GeoMathKit.arrayAdd(C, self.__cTotal)
            self.__sTotal = GeoMathKit.arrayAdd(S, self.__sTotal)
        return self

    def __instance_force(self):
        for tide in self.__include_GravHarType:
            if tide is GravHarType.OceanTide:
                obj = None
                if self.__instance_GravHar[tide] is EnumType.InstanceOceanTide.EOT11a:
                    obj = EOT11a().load()
                elif self.__instance_GravHar[tide] is EnumType.InstanceOceanTide.FES2014:
                    obj = FES2014().load()
                self.GravHarForce[tide] = obj
            if tide is GravHarType.AtmosphereTide:
                obj = None
                if self.__instance_GravHar[tide] is EnumType.InstanceAtmosTide.AtmosBB2003:
                    obj = AtmosBB2003()
                elif self.__instance_GravHar[tide] is EnumType.InstanceAtmosTide.AtmosAOD:
                    obj = AtmosAOD().loadfile()
                self.GravHarForce[tide] = obj
            if tide is GravHarType.PoleTide:
                obj = None
                if self.__instance_GravHar[tide] is EnumType.InstancePoleTide.PoleTideIERS2010:
                    obj = PoleTideIERS2010().load()
                self.GravHarForce[tide] = obj
            if tide is GravHarType.SolidEarthTide:
                obj = None
                if self.__instance_GravHar[tide] is EnumType.InstanceSolidTide.SolidTideIERS2010:
                    obj = SolidTideIERS2010(self.__ep)
                self.GravHarForce[tide] = obj
            if tide is GravHarType.NonTide:
                obj = None
                if self.__instance_GravHar[tide] is EnumType.InstanceNonTide.AODRL06:
                    obj = AODRL06().configure()
                self.GravHarForce[tide] = obj
            if tide is GravHarType.ReferenceModel:
                obj = None
                staticModel = ConfigStaticGravModel().getStaticModel()
                if self.__instance_GravHar[tide] is EnumType.InstanceReferenceModel.RefGravModel:
                    obj = RefGravModel().setModel(staticModel)
                self.GravHarForce[tide] = obj

