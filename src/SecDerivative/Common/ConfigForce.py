from src.ForceModel.ThreeBody import ThreeBody, ThreeBodyPlusJ2
from src.ForceModel.Relativity import RelativityIERS2010
from src.Preference.EnumType import ForceType
import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.Preference.Pre_ForceModel import ThreeBodyConfig
from src.SecDerivative.Common.ConfigForce_GravHar import ConfigGravHar
from src.Frame.PlanetEphemerides import PlanetEphemerides
from src.ForceModel.NonConservative import NonConByAcc
from src.Preference.ForceModelConfig import ForceModelConfig


class ConfigForce:

    def __init__(self, arc: int, ep: PlanetEphemerides):
        self.__forceType = {}
        self.__ep = ep
        self.__arc = arc
        self.__FMConfig = None
        self.__sat = None
        self.__include_ForceType = None

    def configure(self, FMConfig: ForceModelConfig):
        '''force model config'''
        self.__FMConfig = FMConfig
        sat = self.__FMConfig.NonConservativeConfig['sat']
        include_ForceType = self.__FMConfig.include_ForceType.copy()
        self.__sat = [key for key, value in sat.items() if value]
        self.__include_ForceType = [key for key, value in include_ForceType.items() if value]
        self.__instanceForce()
        return self

    def reConfigure(self):
        self.__instanceForce()

    def __instanceForce(self):
        for force in self.__include_ForceType:
            obj = None
            if force == ForceType.GravHar.name:
                obj = ConfigGravHar(self.__ep)
                self.__forceType[force] = obj
            if force == ForceType.Relativity.name:
                obj = RelativityIERS2010().setConstant().setKind()
                self.__forceType[force] = obj
            if force == ForceType.PlanetaryPerturbation.name:
                if ThreeBodyConfig.isJ2:
                    obj = ThreeBodyPlusJ2().configure()
                else:
                    obj = ThreeBody().configure()
                self.__forceType[force] = obj
            if force == ForceType.NonConserveForce.name:
                for sat in self.__sat:
                    obj = NonConByAcc(sat=sat, arcNo=self.__arc).configure()
                    self.__forceType[force + sat.name] = obj

    def getForceType(self):
        return self.__forceType

