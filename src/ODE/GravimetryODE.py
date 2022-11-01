from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.ODE.ComplexODE import ComplexODE, ConfigComplexODE
from src.Preference.EnumType import SingleStepType, MultiStepType
import src.Preference.Pre_ODE as Pre_ODE
from src.Preference.EnumType import TimeFormat, GravHarType
from src.ForceModel.InstanceForceModel import Orbit_var_2nd_diff
from src.ForceModel.VarEqn import VarEqn
from src.Frame.Frame import Frame
import numpy as np
import src.Preference.Pre_Parameterization as Pre_Parameterization
from src.Frame.EOP import EOP


class GravimetryODE:

    def __init__(self):
        self.__t0 = None
        self.__r0 = None
        self.__v0 = None
        self.__force = None
        self.__ParNum = 0
        self.getParNum()
        pass

    def getParNum(self):
        if Pre_Parameterization.TransitionMatrix.isRequired:
            self.__ParNum += Pre_Parameterization.TransitionMatrix.Parameter_Number
        if Pre_Parameterization.Accelerometer.isRequired:
            self.__ParNum += Pre_Parameterization.Accelerometer.Parameter_Number
        if Pre_Parameterization.StokesCoefficients.isRequired:
            self.__ParNum += Pre_Parameterization.StokesCoefficients.Parameter_Number
        return self

    def setInitial(self, t0=None, r0=None, v0=None):
        self.__t0 = t0
        self.__r0 = r0
        self.__v0 = v0
        pv = np.append(r0, v0)

        self.__r0, self.__v0 = VarEqn.getVarEq2ndIni(self.__ParNum, True, True, pv)
        return self

    def setForceModel(self, force: Assemble2ndDerivative):
        self.__force = force.fun1
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
        config = ConfigComplexODE().setOdePar()
        propagator = ComplexODE(config).setSecondDerivative(self.__force).setInitial(self.__t0, self.__r0, self.__v0)
        return propagator.propagate()

