from src.Frame.Frame import Frame
from src.ForceModel.ThreeBody import ThreeBody, ThreeBodyPlusJ2
from src.ForceModel.Relativity import RelativityIERS2010
from src.Preference.EnumType import ForceType, GravHarType
import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.Preference.Pre_ForceModel import ThreeBodyConfig
from src.ForceModel.ConfigGravHar import ConfigGravHar
from src.ForceModel.PlanetEphemerides import PlanetEphemerides
from src.ForceModel.NonConservative import NonConByAcc
from src.ForceModel.GravHar import ClibGravHar
import numpy as np


class ConfigForce:

    def __init__(self, ep: PlanetEphemerides):
        self.__forceType = {}
        self.__ep = ep
        include_ForceType = Pre_ForceModel.include_ForceType.copy()
        self.__include_ForceType = [key for key, value in include_ForceType.items() if value]
        self.__instanceForce()

    def __instanceForce(self):
        for force in self.__include_ForceType:
            obj = None
            if force is ForceType.GravHar:
                obj = ConfigGravHar(self.__ep)
                self.__forceType[force] = obj
            if force is ForceType.Relativity:
                obj = RelativityIERS2010().setConstant().setKind()
                self.__forceType[force] = obj
            if force is ForceType.PlanetaryPerturbation:
                if ThreeBodyConfig.isJ2:
                    obj = ThreeBodyPlusJ2().configure()
                else:
                    obj = ThreeBody().configure()
                self.__forceType[force] = obj
            if force is ForceType.NonConserveForce:
                obj = NonConByAcc().configure()
                self.__forceType[force] = obj

    def getForceType(self):
        return self.__forceType

