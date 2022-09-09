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

    def __init__(self):
        self.__time = None
        self.__pos = None
        self.__vel = None
        self.__ep = None
        self.__acc = np.zeros(3, dtype=float)
        self.__dadp = np.array([[], [], []])
        self.__dadx = np.zeros((3, 3), dtype=float)
        self.ForceType = {}
        self.__instanceForce()

        include_ForceType = Pre_ForceModel.include_ForceType
        self.__include_ForceType = [key for key, value in include_ForceType.items() if value]

    def setTime(self, time: Frame):
        self.__time = time
        self.__ep = PlanetEphemerides().configure().setPlanets().setTime(self.__time)

        self.__calculation()
        return self

    def setPosAndVel(self, pos, vel):
        self.__pos = pos
        self.__vel = vel
        return self

    def __instanceForce(self):
        for force in self.__include_ForceType:
            obj = None
            if force is ForceType.GravHar:
                obj = ConfigGravHar()
                self.ForceType[force] = obj
            if force is ForceType.Relativity:
                obj = RelativityIERS2010().setConstant().setKind().\
                    setSunState(*(self.__ep.getSunPosAndVel(self.__time))).setSatState(self.__pos, self.__vel)
                self.ForceType[force] = obj
            if force is ForceType.PlanetaryPerturbation:
                if ThreeBodyConfig.isJ2:
                    obj = ThreeBodyPlusJ2().setTime(self.__time).\
                        setPlanetPos(self.__ep.getPosByFrame()).setSatPos(self.__pos)
                else:
                    obj = ThreeBody().setTime(self.__time). \
                        setPlanetPos(self.__ep.getPosByFrame()).setSatPos(self.__pos)
                self.ForceType[force] = obj
            if force is ForceType.NonConserveForce:
                obj = NonConByAcc().configure().setTime(self.__time)
                self.ForceType[force] = obj


