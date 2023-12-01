import numpy as np
from tqdm import tqdm
from src.ODE.Common.SingleStepODE import RK4, RK8, RKN
from src.ODE.Common.MultiStepODE import GaussJackson, ABM8, MatrixGaussJackson
from src.Preference.EnumType import SingleStepType, MultiStepType
from src.Preference.Pre_ODE import ODEConfig


class ConfigComplexODE:

    def __init__(self):
        self.stepSize = None
        self.multiStepSize = None
        self.Npoints = None
        self.order = None
        self.SingleStepOde = None
        self.MultiStepOde = None
        self.ODEConfig = None
        self.__ComplexConfig = None
        self.__SingleConfig = None
        self.__MultiConfig = None
        self.__KeplerConfig = None
        self.desc = 'Orbit integration'

    def configure(self, ODEConfig:ODEConfig):
        self.ODEConfig = ODEConfig
        '''config ODE'''
        self.__ComplexConfig = self.ODEConfig.Complex()
        self.__ComplexConfig.__dict__.update(self.ODEConfig.ComplexConfig.copy())
        self.__SingleConfig = self.ODEConfig.SingleStep()
        self.__SingleConfig.__dict__.update(self.ODEConfig.SingleStepConfig.copy())
        self.__MultiConfig = self.ODEConfig.MultiStep()
        self.__MultiConfig.__dict__.update(self.ODEConfig.MultiStepConfig.copy())
        self.__KeplerConfig = self.ODEConfig.KO()
        self.__KeplerConfig.__dict__.update(self.ODEConfig.KOConfig.copy())
        single = self.__ComplexConfig.SingleStepType
        multi = self.__ComplexConfig.MultiStepType
        '''instance ODE'''
        if single == SingleStepType.RK4.name:
            self.SingleStepOde = RK4()
        elif single == SingleStepType.RK8.name:
            self.SingleStepOde = RK8()
        elif single == SingleStepType.RKN.name:
            self.SingleStepOde = RKN()

        if multi == MultiStepType.GaussJackson.name:
            self.MultiStepOde = GaussJackson()
        elif multi == MultiStepType.MatrixGaussJackson.name:
            self.MultiStepOde = MatrixGaussJackson()
        elif multi == MultiStepType.ABM8.name:
            self.MultiStepOde = ABM8()
        return self

    def setOdePar(self):
        singleStepSize = self.__SingleConfig.stepsize
        multiStepSize = self.__MultiConfig.stepsize
        assert singleStepSize == multiStepSize
        self.stepSize = singleStepSize
        self.Npoints = self.__MultiConfig.Npoints
        self.order = self.__MultiConfig.order
        if isinstance(self.MultiStepOde, GaussJackson) or \
                isinstance(self.MultiStepOde, MatrixGaussJackson):
            self.MultiStepOde.setOrder(order=self.order).\
                setIterArgs(iterNum=self.__MultiConfig.iterNum, rltot=self.__MultiConfig.rltot)
        return self


class ComplexODE:

    def __init__(self, config: ConfigComplexODE):
        self.__desc = config.desc
        self.__SingleStepOde = config.SingleStepOde.configure(ODEConfig=config.ODEConfig).setRecord()
        self.__MultiStepOde = config.MultiStepOde.configure(ODEConfig=config.ODEConfig)
        self.__stepSize = config.stepSize
        self.__Npoints = None
        self.__order = config.order
        self.__secDer = None

        pass

    def setSecondDerivative(self, secDer, ArcLen):
        """
        set the force function for orbit integration.
        This function is exposed to the user, while it should be consistent with the config.
        Notice: it must be immediately used after config!
        :param secDer: father class of ForceModel, e.g., ForcesDoubleSat, ForcesSingleSat
        :return:
        """
        '''set force model'''
        self.__Npoints = ArcLen
        self.__secDer = secDer
        self.__SingleStepOde.setForce(secDer)
        self.__MultiStepOde.setForce(secDer)
        return self

    def setInitial(self, t0=None, r0=None, v0=None):
        """
        Assign the initial value for orbit and variational equations propagators
        Notice: it has to be used after 'setForceModel'
        1. rA, vA
        2. rB, vB
        3. rA+rB, vA+vB

        :param t0: time epoch  t0
        :param r0: pos vector at t0
        :param v0: vel at t0
        :return:
        """

        '''Configuration for starter'''
        self.__SingleStepOde.setInitial(r0, v0, t0)

        '''Start-up: get initial points'''
        x1, r1, v1 = self.__SingleStepOde.propagate(int(self.__order / 2), -self.__stepSize)
        x2, r2, v2 = self.__SingleStepOde.propagate(int(self.__order / 2), self.__stepSize)
        x1.pop(0)
        r1.pop(0)
        v1.pop(0)
        x1.reverse()
        r1.reverse()
        v1.reverse()
        iniX = x1 + x2
        iniR = r1 + r2
        iniV = v1 + v2

        self.__MultiStepOde.setInitial(iniX.copy(), iniR.copy(), iniV.copy())
        return self

    def propagate(self):
        """
        Orbit integration by given time series.
        :param Nstep: include t0, e.g. if Nstep = 5, we have [t0, t1, t2, t3, t4]
        :return:
        """

        h = self.__stepSize
        assert self.__Npoints > self.__order / 2 + 1

        '''Step by Step'''
        dataOut = self.__MultiStepOde.propagate_one_step(h, isFirstStep=True, dataIn=None)

        epoch = []
        resR, resV = [], []
        '''record'''
        epoch += list(dataOut[0])
        resR += list(dataOut[1])
        resV += list(dataOut[2])

        for i in tqdm(range(int(self.__Npoints - self.__order / 2 - 1)), desc=self.__desc):
        # for i in range(int(self.__Npoints - self.__order / 2 - 1)):
            dataOut = self.__MultiStepOde.propagate_one_step(h, isFirstStep=False, dataIn=dataOut)
            epoch.append(dataOut[0][-1])
            resR.append(dataOut[1][-1])
            resV.append(dataOut[2][-1])

        return np.array(epoch), np.array(resR), np.array(resV)

