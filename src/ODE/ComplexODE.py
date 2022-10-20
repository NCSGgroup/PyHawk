import numpy as np
import json
from tqdm import tqdm
from src.ODE.SingleStepODE import SingleStepOde, RK4, RK8, RKN, MatrixRKN
from src.ODE.MultiStepODE import MultiStepOde, GaussJackson, ABM8, MatrixGaussJackson
import src.Preference.Pre_ODE as Pre_ODE
from src.Preference.EnumType import SingleStepType, MultiStepType


class ConfigComplexODE:

    def __init__(self, single: SingleStepType, multi: MultiStepType):
        self.stepSize = None
        self.multiStepSize = None
        self.Npoints = None
        self.order = None
        self.SingleStepOde = None
        self.MultiStepOde = None
        self.desc = 'Orbit integration'

        if single is SingleStepType.RK4:
            self.SingleStepOde = RK4()
        elif single is SingleStepType.RK8:
            self.SingleStepOde = RK8()
        elif single is SingleStepType.RKN:
            self.SingleStepOde = RKN()
        elif single is SingleStepType.MatrixRKN:
            self.SingleStepOde = MatrixRKN()

        if multi is MultiStepType.GaussJackson:
            self.MultiStepOde = GaussJackson()
        elif multi is MultiStepType.MatrixGaussJackson:
            self.MultiStepOde = MatrixGaussJackson()
        elif multi is MultiStepType.ABM8:
            self.MultiStepOde = ABM8()

    def setOdePar(self, singleStepSize=Pre_ODE.SingleStepConfig.stepsize,
                  multiStepSize=Pre_ODE.MultiStepConfig.stepsize,
                  Npoints=Pre_ODE.MultiStepConfig.Npoints,
                  order=Pre_ODE.MultiStepConfig.order):
        assert singleStepSize == multiStepSize
        self.stepSize = singleStepSize
        self.Npoints = Npoints
        self.order = order
        if isinstance(self.MultiStepOde, GaussJackson) or \
                isinstance(self.MultiStepOde, MatrixGaussJackson):
            self.MultiStepOde.setOrder().setIterArgs()
        return self


class ComplexODE:

    def __init__(self, config: ConfigComplexODE):
        self.__desc = config.desc
        self.__SingleStepOde = config.SingleStepOde.setRecord()
        self.__MultiStepOde = config.MultiStepOde
        self.__stepSize = config.stepSize
        self.__Npoints = config.Npoints
        self.__order = config.order

        pass

    def setSecondDerivative(self, force):
        """
        set the force function for orbit integration.
        This function is exposed to the user, while it should be consistent with the config.
        Notice: it must be immediately used after config!
        :param force: father class of ForceModel, e.g., ForcesDoubleSat, ForcesSingleSat
        :return:
        """
        '''set force model'''
        self.__force = force
        self.__SingleStepOde.setForce(force)
        self.__MultiStepOde.setForce(force)
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
            dataOut = self.__MultiStepOde.propagate_one_step(h, isFirstStep=False, dataIn=dataOut)
            epoch.append(dataOut[0][-1])
            resR.append(dataOut[1][-1])
            resV.append(dataOut[2][-1])

        return np.array(epoch), np.array(resR), np.array(resV)

