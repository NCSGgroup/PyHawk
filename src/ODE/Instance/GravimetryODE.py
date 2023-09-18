from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.ODE.Common.ComplexODE import ComplexODE, ConfigComplexODE
from src.Preference.EnumType import SingleStepType, MultiStepType
from src.SecDerivative.MatrixVarEqn import MatrixVarEqn
import numpy as np
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_ODE import ODEConfig


class GravimetryODE:

    def __init__(self):
        self.__t0 = None
        self.__r0 = None
        self.__v0 = None
        self.__secDer = None
        self.__ParNum = 0
        self.__isSingle = True
        self.ParameterConfig = None
        self.ODEConfig = None
        self._ArcLen = None
        self.TransitionMatrixConfig = None
        self.AccelerometerConfig = None
        self.StokesCoefficientsConfig = None
        pass

    def configure(self, ODEConfig: ODEConfig, ParameterConfig: ParameterConfig):
        self.ODEConfig = ODEConfig
        self.ParameterConfig = ParameterConfig

        '''config TransitionMatrix'''
        self.TransitionMatrixConfig = self.ParameterConfig.TransitionMatrix()
        self.TransitionMatrixConfig.__dict__.update(self.ParameterConfig.TransitionMatrixConfig.copy())
        '''config AccelerometerConfig'''
        self.AccelerometerConfig = self.ParameterConfig.Accelerometer()
        self.AccelerometerConfig.__dict__.update(self.ParameterConfig.AccelerometerConfig.copy())
        '''config StokesCoefficientsConfig'''
        self.StokesCoefficientsConfig = self.ParameterConfig.StokesCoefficients()
        self.StokesCoefficientsConfig.__dict__.update(self.ParameterConfig.StokesCoefficientsConfig.copy())
        return self

    def setParNum(self):
        if self.TransitionMatrixConfig.isRequired:
            self.__ParNum += self.TransitionMatrixConfig.Parameter_Number
        if self.AccelerometerConfig.isRequired:
            self.__ParNum += self.AccelerometerConfig.Parameter_Number
        if self.StokesCoefficientsConfig.isRequired:
            self.__ParNum += self.StokesCoefficientsConfig.Parameter_Number
        return self

    def setInitial(self, t0=None, r0=None, v0=None):
        self.__t0 = t0
        self.__r0 = r0
        self.__v0 = v0

        if np.shape(r0) == np.shape(v0) == (2, 3):
            self.__isSingle = False
            pv1 = np.append(r0[0], v0[0])
            pv2 = np.append(r0[1], v0[1])
            rA, vA = MatrixVarEqn.getVarEq2ndIni(self.__ParNum, self.TransitionMatrixConfig.isRequired, True,
                                                 pv1)
            rB, vB = MatrixVarEqn.getVarEq2ndIni(self.__ParNum, self.TransitionMatrixConfig.isRequired, True,
                                                 pv2)
            self.__r0, self.__v0 = np.vstack((rA, rB)), np.vstack((vA, vB))
            pass
        elif len(r0) == len(v0) == 3:
            self.__isSingle = True
            pv = np.append(r0, v0)
            r, v = MatrixVarEqn.getVarEq2ndIni(self.__ParNum, self.TransitionMatrixConfig.isRequired, True, pv)
            self.__r0, self.__v0 = r, v

        return self

    def set2ndDerivative(self, secDer: Assemble2ndDerivative):
        self._ArcLen = secDer.getArcLen()
        self.__secDer = secDer.secDerivative
        return self

    # def __forceModel(self, t, r, v):
    #     self.__fr = self.__fr.setTime(t, TimeFormat.GPS_second)
    #     self.__force = self.__force.setPosAndVel(r, v).setTime(self.__fr)
    #     return self.__force.getAcceleration()
    #
    # def __MatrixforceModel(self, t, r, v):
    #     self.__fr = self.__fr.setTime(t, TimeFormat.GPS_second)
    #     if len(r) == 2 and len(v) == 2:
    #         acc1 = self.__force.setPosAndVel(r[0], v[0]).setTime(self.__fr).getAcceleration()
    #         acc2 = self.__force.setPosAndVel(r[1], v[1]).setTime(self.__fr).getAcceleration()
    #         acc = np.vstack((acc1, acc2))
    #         return acc
    #     else:
    #         acc = self.__force.setPosAndVel(r, v).setTime(self.__fr).getAcceleration()
    #         return acc

    def propagate(self):
        config = ConfigComplexODE().configure(ODEConfig=self.ODEConfig).setOdePar()
        propagator = ComplexODE(config).setSecondDerivative(self.__secDer, ArcLen=self._ArcLen - 4).\
            setInitial(self.__t0, self.__r0, self.__v0)
        return propagator.propagate()

