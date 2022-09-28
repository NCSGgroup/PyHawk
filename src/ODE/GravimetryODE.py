from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.ODE.ComplexODE import ComplexODE, ConfigComplexODE
from src.Preference.EnumType import SingleStepType, MultiStepType
import src.Preference.Pre_ODE as Pre_ODE
from src.Preference.EnumType import TimeFormat, GravHarType
from src.Frame.Frame import Frame
import numpy as np
from src.Frame.EOP import EOP


class GravimetryODE:

    def __init__(self, fr: Frame):
        self.__fr = fr
        self.__t0 = None
        self.__r0 = None
        self.__v0 = None
        self.__force = None
        pass

    def setInitial(self, t0=None, r0=None, v0=None):
        self.__t0 = t0
        self.__r0 = r0
        self.__v0 = v0
        return self

    def setForceModel(self, force: Assemble2ndDerivative):
        self.__force = force
        return self

    def forceModel(self, t, r, v):
        self.__fr = self.__fr.setTime(t, TimeFormat.GPS_second)
        self.__force = self.__force.setPosAndVel(r, v).setTime(self.__fr)
        return self.__force.getAcceleration()

    def propagate(self):
        config = ConfigComplexODE(SingleStepType.RKN, MultiStepType.GaussJackson).setOdePar()
        propagator = ComplexODE(config).setSecondDerivative(self.forceModel).setInitial(self.__t0, self.__r0, self.__v0)
        return propagator.propagate()

