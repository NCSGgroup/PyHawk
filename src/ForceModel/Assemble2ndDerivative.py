from src.Frame.Frame import Frame
from src.ForceModel.ThreeBody import AbsThreeBody
from src.ForceModel.Relativity import AbsRelativity
from src.ForceModel.ConfigForce import ConfigForce
from src.ForceModel.ConfigGravHar import ConfigGravHar
from src.Auxilary.PlanetEphemerides import PlanetEphemerides
from src.ForceModel.NonConservative import AbsNonConservative
from src.Auxilary.SortCS import SortCS
import src.Preference.Pre_Parameterization as Pre_Parameterization
import numpy as np


class Assemble2ndDerivative:

    def __init__(self):
        self.__time = None
        self.__pos = None
        self.__vel = None
        self.__ep = PlanetEphemerides().configure().setPlanets()
        self.__sort = SortCS(Pre_Parameterization.StokesCoefficients.SortMethod,
                             Pre_Parameterization.StokesCoefficients.MaxDegree,
                             Pre_Parameterization.StokesCoefficients.MinDegree)
        self.ForceType = ConfigForce(self.__ep).getForceType()

        self.__isExistDaDx = False
        self.__isExistDaDp = False
        self.__acc = np.zeros(3, dtype=float)
        self.__dadp = np.array([[], [], []])
        self.__dadx = np.zeros((3, 3), dtype=float)

    def __reset(self):
        self.__isExistDaDx = False
        self.__isExistDaDp = False
        self.__acc = np.zeros(3, dtype=float)
        self.__dadp = np.array([[], [], []])
        self.__dadx = np.zeros((3, 3), dtype=float)
        pass

    def setTime(self, time: Frame):
        self.__reset()
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
                c, s = obj.setTime(self.__time).getTotalCS()
                if obj.isIncludeEarth:
                    c[0:3], s[0:3] = 0, 0
                if not obj.isIncludeSecOrder:
                    c[3:6], s[3:6] = 0, 0
                if not obj.isIncludeThirdOrder:
                    c[6:10], s[6:10] = 0, 0
                cgh = obj.ClibGravHar.setPar(c, s, self.__pos)
                self.__acc += cgh.getAcceleration()

                if Pre_Parameterization.TransitionMatrix.isRequired:
                    res = cgh.getDaDx()
                    self.__isExistDaDx = True
                    self.__dadx += self.__time.dadrITRS2GCRS(res, self.__time.getRotationMatrix)

                if Pre_Parameterization.StokesCoefficients.isRequired:
                    du2c, du2s = cgh.getDaDp(degree_max=Pre_Parameterization.StokesCoefficients.MaxDegree,
                                             degree_min=Pre_Parameterization.StokesCoefficients.MinDegree)
                    res = np.array([self.__sort.sort(du2c[0], du2s[0]),
                                    self.__sort.sort(du2c[1], du2s[1]),
                                    self.__sort.sort(du2c[2], du2s[2])])
                    res = self.__time.AccITRS2GCRS(res, self.__time.getRotationMatrix)
                    self.__isExistDaDp = True
                    self.__dadp = np.hstack((self.__dadp, res))
            elif isinstance(obj, AbsRelativity):
                obj = obj.setSunState(*(self.__ep.getSunPosAndVel(self.__time))).setSatState(self.__pos, self.__vel)
                self.__acc += obj.getAcceleration()
            elif isinstance(obj, AbsNonConservative):
                obj = obj.setTime(self.__time)
                self.__acc += obj.getAcceleration()

                if Pre_Parameterization.Accelerometer.isRequired:
                    self.__isExistDaDp = True
                    self.__dadp = np.hstack((self.__dadp,  obj.getDaDp()))
            elif isinstance(obj, AbsThreeBody):
                obj = obj.setTime(self.__time).setPlanetPos(self.__ep.getPosByFrame()).setSatPos(self.__pos)
                self.__acc += obj.getAcceleration()
                if Pre_Parameterization.TransitionMatrix.isRequired:
                    self.__isExistDaDx = True
                    self.__dadx += obj.getDaDx()

    def getAcceleration(self):
        if not bool(self.ForceType):
            return None
        return self.__acc

    def getDaDx(self):
        if not self.__isExistDaDx:
            return None
        return self.__dadx

    def getDaDp(self):
        if not self.__isExistDaDp:
            return None
        return self.__dadp

