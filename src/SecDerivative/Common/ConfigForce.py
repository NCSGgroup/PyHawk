from src.ForceModel.ThreeBody import ThreeBody, ThreeBodyPlusJ2
from src.ForceModel.Relativity import RelativityIERS2010
from src.Preference.EnumType import ForceType
import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.Preference.Pre_ForceModel import ThreeBodyConfig
from src.SecDerivative.Common.ConfigForce_GravHar import ConfigGravHar
from src.Frame.PlanetEphemerides import PlanetEphemerides
from src.ForceModel.NonConservative import NonConByAcc


class ConfigForce:

    def __init__(self, ep: PlanetEphemerides):
        self.__forceType = {}
        self.__ep = ep
        sat = Pre_ForceModel.NonConservative.sat.copy()
        self.__sat = [key for key, value in sat.items() if value]
        include_ForceType = Pre_ForceModel.include_ForceType.copy()
        self.__include_ForceType = [key for key, value in include_ForceType.items() if value]
        self.__instanceForce()

    def reConfigure(self):
        self.__instanceForce()

    def __instanceForce(self):
        for force in self.__include_ForceType:
            obj = None
            if force is ForceType.GravHar:
                obj = ConfigGravHar(self.__ep)
                self.__forceType[force.name] = obj
            if force is ForceType.Relativity:
                obj = RelativityIERS2010().setConstant().setKind()
                self.__forceType[force.name] = obj
            if force is ForceType.PlanetaryPerturbation:
                if ThreeBodyConfig.isJ2:
                    obj = ThreeBodyPlusJ2().configure()
                else:
                    obj = ThreeBody().configure()
                self.__forceType[force.name] = obj
            if force is ForceType.NonConserveForce:
                for sat in self.__sat:
                    obj = NonConByAcc(sat=sat).configure()
                    self.__forceType[force.name + sat.name] = obj

    def getForceType(self):
        return self.__forceType

