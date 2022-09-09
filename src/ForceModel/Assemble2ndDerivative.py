from src.Frame.Frame import Frame
from src.ForceModel.ThreeBody import ThreeBody, ThreeBodyPlusJ2
from src.ForceModel.Relativity import RelativityIERS2010
from src.ForceModel.ConfigForce import ConfigForce
from src.ForceModel.ConfigGravHar import ConfigGravHar
from src.ForceModel.PlanetEphemerides import PlanetEphemerides
from src.ForceModel.NonConservative import NonConByAcc
from src.ForceModel.GravHar import ClibGravHar
from src.Auxilary.SortCS import SortCS
import numpy as np


class AssembleForce:

    def __init__(self):
        self.__time = None
        self.__pos = None
        self.__vel = None

        self.__ep = PlanetEphemerides().configure().setPlanets()
        self.__cgh = ClibGravHar().setConstant()
        self.__sort = SortCS()
        self.ForceType = ConfigForce().ForceType

        self.__acc = np.zeros(3, dtype=float)
        self.__dadp = np.array([[], [], []])
        self.__dadx = np.zeros((3, 3), dtype=float)

    def setTime(self, time: Frame):
        self.__time = time
        self.__ep = self.__ep.setTime(self.__time)
        self.__calculation()
        return self

    def setPosAndVel(self, pos, vel):
        self.__pos = pos
        self.__vel = vel
        return self

    def __calculation(self):
        for obj in self.ForceType.values():
            if isinstance(obj, ConfigGravHar):
                c, s = obj.setTime(self.__time).getSumCS()
                self.__acc += self.__cgh.setPar(c, s, self.__pos).getAcceleration()
                self.__dadx += self.__cgh.setPar(c, s, self.__pos).getDaDx()
                du2c, du2s = self.__cgh.setPar(c, s, self.__pos).getDaDp()
                res = np.array([self.__sort.sort(du2c[0], du2s[0]),
                                self.__sort.sort(du2c[1], du2s[1]),
                                self.__sort.sort(du2c[2], du2s[2]),])
                res = self.__time.AccITRS2GCRS(res, self.__time.getRotationMatrix)

                # self.__dadp = np.hstack((self.__dadp, self.__cgh.setPar(c, s, self.__pos).getDaDp()))
            elif isinstance(obj, RelativityIERS2010):
                obj = obj.setSunState(*(self.__ep.getSunPosAndVel(self.__time))).setSatState(self.__pos, self.__vel)
                self.__acc += obj.getAcceleration()
            elif isinstance(obj, NonConByAcc):
                obj = obj.setTime(self.__time)
                self.__acc += obj.getAcceleration()
                # self.__dadp += obj.getDaDp()
            elif isinstance(obj, ThreeBodyPlusJ2) or isinstance(obj, ThreeBody):
                obj = obj.setTime(self.__time).setPlanetPos(self.__ep.getPosByFrame()).setSatPos(self.__pos)
                self.__acc += obj.getAcceleration()
                self.__dadx += obj.getDaDx()

    def getAcceleration(self):
        return self.__acc

    def getDaDx(self):
        return self.__dadx

    def getDaDp(self):
        return self.__dadp

