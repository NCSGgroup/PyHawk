from src.Frame.Frame import Frame
from src.SecDerivative.Common.ConfigForce import ConfigForce
from src.SecDerivative.Common.ConfigForce_GravHar import ConfigGravHar
from src.Frame.PlanetEphemerides import PlanetEphemerides
import src.Preference.EnumType as EnumType
from src.Auxilary.SortCS import SortCS
import numpy as np
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_Frame import FrameConfig


class Assemble2ndDerivative:

    def __init__(self):
        self.__time = None
        self.__pos = None
        self.__vel = None
        self.__C = None
        self.__S = None
        self._arc = None
        self.__Sat = EnumType.SatID.A
        self.__sort = None
        self.__isExistDaDx = False
        self.__isExistDaDp = False
        self.__externalEOP = None
        self.__ArcLen = None
        '''force model cofig'''
        self.configForce = None
        self.ForceType = None
        '''PlanetEphemerides'''
        self.__ep = None
        '''init res'''
        self.__acc = np.zeros(3, dtype=float)
        self.__dadp = np.array([[], [], []])
        self.__dadx = np.zeros((3, 3), dtype=float)
        self.__FMConfig = None
        self._ParameterConfig = None
        self._TransitionMatrixConfig = None
        self._AccelerometerConfig = None
        self._StokesCoefficientsConfig = None
        pass

    def configure(self, FMConfig: ForceModelConfig, ParameterConfig: ParameterConfig, FrameConfig: FrameConfig):
        self._ParameterConfig = ParameterConfig
        '''config TransitionMatrix'''
        self._TransitionMatrixConfig = self._ParameterConfig.TransitionMatrix()
        self._TransitionMatrixConfig.__dict__.update(self._ParameterConfig.TransitionMatrixConfig.copy())
        '''config AccelerometerConfig'''
        self._AccelerometerConfig = self._ParameterConfig.Accelerometer()
        self._AccelerometerConfig.__dict__.update(self._ParameterConfig.AccelerometerConfig.copy())
        '''config StokesCoefficientsConfig'''
        self._StokesCoefficientsConfig = self._ParameterConfig.StokesCoefficients()
        self._StokesCoefficientsConfig.__dict__.update(self._ParameterConfig.StokesCoefficientsConfig.copy())
        '''config sort'''
        self.__sort = SortCS(method=self._StokesCoefficientsConfig.SortMethod,
                             degree_max=self._StokesCoefficientsConfig.MaxDegree,
                             degree_min=self._StokesCoefficientsConfig.MinDegree)
        self.__ep = PlanetEphemerides().configure(FMConfig=FMConfig, FrameConfig=FrameConfig).setPlanets()
        self.configForce = ConfigForce(arc=self._arc, ep=self.__ep).configure(FMConfig=FMConfig)
        self.ForceType = self.configForce.getForceType()
        return self

    def reConfigure(self):
        self.configForce.reConfigure()
        return self

    def __reset(self):
        self.__isExistDaDx = False
        self.__isExistDaDp = False
        self.__acc = np.zeros(3, dtype=float)
        self.__dadp = np.array([[], [], []])
        self.__dadx = np.zeros((3, 3), dtype=float)
        pass

    def setTime(self, time: Frame):
        self.__time = time
        self.__ep = self.__ep.setTime(self.__time).getPosByFrame()
        return self

    def setPosAndVel(self, pos, vel):
        self.__pos = pos
        self.__vel = vel
        return self

    def getCS(self):
        for obj in self.ForceType.values():
            if isinstance(obj, ConfigGravHar):
                self.__C, self.__S = obj.setEOP(self.__externalEOP).setTime(self.__time).getTotalCS()
        return self

    def ChangeSat(self, sat: EnumType.SatID):
        self.__Sat = sat
        return self

    def calculation(self):
        self.__reset()
        for item in self.ForceType:
            obj = self.ForceType[item]
            if item == EnumType.ForceType.GravHar.name:
                rm = self.__time.getRotationMatrix
                pos = self.__pos
                if obj.InputPosCoordinate == EnumType.Coordinate.GCRS.name:
                    pos = self.__time.PosGCRS2ITRS(np.array(self.__pos.copy()).astype(float), rm)
                if len(obj.setZero) != 0:
                    for index in obj.setZero:
                        start = int(index * (index + 1) / 2)
                        end = int((index + 1) * (index + 2) / 2)
                        self.__C[start: end] = 0.
                        self.__S[start: end] = 0.
                cgh = obj.ClibGravHar.setPar(self.__C, self.__S, pos)

                acc = cgh.getAcceleration()
                if obj.OutputCoordinate == EnumType.Coordinate.GCRS.name:
                    acc = self.__time.AccITRS2GCRS(acc, rm).flatten()
                self.__acc += acc
                if self._TransitionMatrixConfig.isRequired:
                    res = cgh.getDaDx()
                    self.__isExistDaDx = True
                    self.__dadx += self.__time.dadrITRS2GCRS(res, self.__time.getRotationMatrix)
                if self._StokesCoefficientsConfig.isRequired:
                    du2c, du2s = cgh.getDaDp(degree_max=self._StokesCoefficientsConfig.MaxDegree,
                                             degree_min=self._StokesCoefficientsConfig.MinDegree)
                    res = np.array([self.__sort.sort(du2c[0], du2s[0]),
                                    self.__sort.sort(du2c[1], du2s[1]),
                                    self.__sort.sort(du2c[2], du2s[2])])
                    res = self.__time.AccITRS2GCRS(res, self.__time.getRotationMatrix)
                    self.__isExistDaDp = True
                    self.__dadp = np.hstack((self.__dadp, res))
            elif item == EnumType.ForceType.Relativity.name:
                obj = obj.setSunState(*(self.__ep.getSunPosAndVel(self.__time))) \
                    .setSatState(self.__pos.copy(), self.__vel.copy())
                self.__acc += obj.getAcceleration()
            elif item == EnumType.ForceType.PlanetaryPerturbation.name:
                obj = obj.setTime(self.__time).setPlanetPos(self.__ep.getPosPlanets()).setSatPos(self.__pos.copy())
                self.__acc += obj.getAcceleration()
                if self._TransitionMatrixConfig.isRequired:
                    self.__isExistDaDx = True
                    self.__dadx += obj.getDaDx()
            elif item == (EnumType.ForceType.NonConserveForce.name + self.__Sat.name):
                obj = obj.setTime(self.__time)
                self.__acc += obj.getAcceleration()
                if self._AccelerometerConfig.isRequired:
                    self.__isExistDaDp = True
                    self.__dadp = np.hstack((self.__dadp, obj.getDaDp()))

        return self
        # for obj in self.ForceType.values():
        #     if isinstance(obj, ConfigGravHar):
        #         rm = self.__time.getRotationMatrix
        #         pos = self.__pos
        #         if obj.InputPosCoordinate is EnumType.Coordinate.GCRS:
        #             pos = self.__time.PosGCRS2ITRS(np.array(self.__pos.copy()).astype(float), rm)
        #         if len(obj.setZero) != 0:
        #             for index in obj.setZero:
        #                 start = int(index * (index + 1) / 2)
        #                 end = int((index + 1) * (index + 2) / 2)
        #                 self.__C[start: end] = 0.
        #                 self.__S[start: end] = 0.
        #         cgh = obj.ClibGravHar.setPar(self.__C, self.__S, pos)
        #
        #         acc = cgh.getAcceleration()
        #         if obj.OutputCoordinate is EnumType.Coordinate.GCRS:
        #             acc = self.__time.AccITRS2GCRS(acc, rm).flatten()
        #         self.__acc += acc
        #         if Pre_Parameterization.TransitionMatrix.isRequired:
        #             res = cgh.getDaDx()
        #             self.__isExistDaDx = True
        #             self.__dadx += self.__time.dadrITRS2GCRS(res, self.__time.getRotationMatrix)
        #         if Pre_Parameterization.StokesCoefficients.isRequired:
        #             du2c, du2s = cgh.getDaDp(degree_max=Pre_Parameterization.StokesCoefficients.MaxDegree,
        #                                      degree_min=Pre_Parameterization.StokesCoefficients.MinDegree)
        #             res = np.array([self.__sort.sort(du2c[0], du2s[0]),
        #                             self.__sort.sort(du2c[1], du2s[1]),
        #                             self.__sort.sort(du2c[2], du2s[2])])
        #             res = self.__time.AccITRS2GCRS(res, self.__time.getRotationMatrix)
        #             self.__isExistDaDp = True
        #             self.__dadp = np.hstack((self.__dadp, res))
        #     elif isinstance(obj, AbsRelativity):
        #         obj = obj.setSunState(*(self.__ep.getSunPosAndVel(self.__time)))\
        #             .setSatState(self.__pos.copy(), self.__vel.copy())
        #         self.__acc += obj.getAcceleration()
        #     elif isinstance(obj, AbsNonConservative):
        #         obj = obj.setTime(self.__time)
        #         self.__acc += obj.getAcceleration()
        #         if Pre_Parameterization.Accelerometer.isRequired:
        #             self.__isExistDaDp = True
        #             self.__dadp = np.hstack((self.__dadp, obj.getDaDp()))
        #     elif isinstance(obj, AbsThreeBody):
        #         obj = obj.setTime(self.__time).setPlanetPos(self.__ep.getPosPlanets()).setSatPos(self.__pos.copy())
        #         self.__acc += obj.getAcceleration()
        #         if Pre_Parameterization.TransitionMatrix.isRequired:
        #             self.__isExistDaDx = True
        #             self.__dadx += obj.getDaDx()

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

    def secDerivative(self, t, r, v):
        pass

    def setEOP(self, xp, yp):
        self.__externalEOP = [xp, yp]
        return self

    def getArcLen(self):
        return self.__ArcLen