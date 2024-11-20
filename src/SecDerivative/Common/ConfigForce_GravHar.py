import os
import sys
sys.path.append('../')
from src.ForceModel.AtmosTide import AtmosBB2003, AtmosAOD
from src.ForceModel.NonTide import AODRL06
from src.ForceModel.OceanTide import EOT11a, FES2014
from src.ForceModel.PoleTide import PoleTideIERS2010
from src.ForceModel.SolidTide import SolidTideIERS2010
from src.ForceModel.RefGravModel import RefGravModel
from src.Preference.EnumType import GravHarType
from src.ForceModel.BaseGravCS import BaseGravCS
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Frame.PlanetEphemerides import PlanetEphemerides
import src.Preference.EnumType as EnumType
from src.ForceModel.GravHar import ClibGravHar
from src.Preference.Pre_ForceModel import ForceModelConfig


class ConfigGravHar:

    def __init__(self, ep: PlanetEphemerides):
        self.__include_GravHarType = None
        self.__isFirst = True
        self.__sTotal = None
        self.__cTotal = None
        self.__time = None
        self.__FMConfig = None
        self.setZero = None
        self.InputPosCoordinate = None
        self.OutputCoordinate = None
        self.__instance_GravHar = None
        self.__ep = ep
        self.GravHarForce = {}
        self._GravHarConfig = None
        pass

    def configure(self, FMConfig: ForceModelConfig):
        '''force model config'''
        self.__FMConfig = FMConfig
        self._GravHarConfig = self.__FMConfig.GravHar()
        self._GravHarConfig.__dict__.update(self.__FMConfig.GravHarConfig.copy())
        self.setZero = self._GravHarConfig.setZero.copy()
        self.InputPosCoordinate = self._GravHarConfig.PosCoordinate
        self.OutputCoordinate = self._GravHarConfig.ACCCoordinate
        '''include force'''
        include_GravHarType = self.__FMConfig.include_GravHarType.copy()
        instance_GravHar = self.__FMConfig.instance_GravHar.copy()
        self.__include_GravHarType = [key for key, value in include_GravHarType.items() if value]
        self.__instance_GravHar = instance_GravHar
        '''instance clib gravhar'''
        self.ClibGravHar = ClibGravHar().configure(self.__FMConfig)
        self.__instance_force()
        return self

    def __reset(self):
        self.__isFirst = True
        self.__sTotal = None
        self.__cTotal = None

    def setTime(self, time: Frame):
        self.__reset()
        self.__time = time
        for obj in self.GravHarForce.values():
            if isinstance(obj, PoleTideIERS2010):
                obj = obj.setEOP(self.__externalEOP)
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
            if tide == GravHarType.OceanTide.name:
                obj = None
                if self.__instance_GravHar[tide] == EnumType.InstanceOceanTide.EOT11a.name:
                    obj = EOT11a().configure(self.__FMConfig).speedUp().load()
                elif self.__instance_GravHar[tide] == EnumType.InstanceOceanTide.FES2014.name:
                    obj = FES2014().configure(self.__FMConfig).speedUp().load()
                self.GravHarForce[tide] = obj
            if tide == GravHarType.AtmosphereTide.name:
                obj = None
                if self.__instance_GravHar[tide] == EnumType.InstanceAtmosTide.AtmosBB2003.name:
                    obj = AtmosBB2003().configure(self.__FMConfig)
                elif self.__instance_GravHar[tide] == EnumType.InstanceAtmosTide.AtmosAOD.name:
                    obj = AtmosAOD().configure(self.__FMConfig).loadfile()
                self.GravHarForce[tide] = obj
            if tide == GravHarType.PoleTide.name:
                obj = None
                if self.__instance_GravHar[tide] == EnumType.InstancePoleTide.PoleTideIERS2010.name:
                    obj = PoleTideIERS2010().configure(self.__FMConfig).load()
                self.GravHarForce[tide] = obj
            if tide == GravHarType.SolidEarthTide.name:
                obj = None
                if self.__instance_GravHar[tide] == EnumType.InstanceSolidTide.SolidTideIERS2010.name:
                    obj = SolidTideIERS2010(self.__ep).configure(self.__FMConfig)
                self.GravHarForce[tide] = obj
            if tide == GravHarType.NonTide.name:
                obj = None
                if self.__instance_GravHar[tide] == EnumType.InstanceNonTide.AODRL06.name:
                    obj = AODRL06().configure(self.__FMConfig)
                self.GravHarForce[tide] = obj
            if tide == GravHarType.ReferenceModel.name:
                obj = None
                obj = RefGravModel().configure(self.__FMConfig)
                staticModel = obj.getStaticModel()
                if self.__instance_GravHar[tide] == EnumType.InstanceReferenceModel.RefGravModel.name:
                    obj = obj.setModel(staticModel)
                self.GravHarForce[tide] = obj

    def setEOP(self, externalEOP):
        self.__externalEOP = externalEOP
        return self