from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat, GravHarType
from src.ForceModel.Assemble2ndDerivative import AssembleForce
from src.ODE.ComplexODE import ComplexODE, ConfigComplexODE
from src.Preference.EnumType import SingleStepType, MultiStepType
from src.ForceModel.MainPropagator import PropagatorConfig, Propagator
import json
import numpy as np


def demo1():
    with open('../setting/GraceTestForceModel_A.json') as load_f:
        configA = json.load(load_f)
    fr = Frame(EOP(True).load())

    '''sat state'''
    posA = np.array([-4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06])
    velA = np.array([3000, 5000, 6000])
    posB = np.array([-4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06])
    velB = np.array([3000, 5000, 6000])
    x0 = 504878405
    h = 30
    Nstep = 720 * 2

    '''sat force'''
    fr.setTime(504878405, TimeFormat.GPS_second)
    modelA = AssembleForce()
    # modelA = modelA.setPosAndVel(posA, velA).setTime(fr)
    pc = PropagatorConfig().configure(configA)
    modelB = AssembleForce()
    # modelB = modelB.setPosAndVel(posA, velA).setTime(fr)
    # print(modelA.getAcceleration())
    # print(modelB.getAcceleration())

    '''ode'''
    config = ConfigComplexODE(SingleStepType.RKN, MultiStepType.GaussJackson).setOdePar()
    propagator = ComplexODE(config).setSecondDerivative(modelA).setInitial(x0, posA, velA)
    X_GJ, R_GJ, v = propagator.propagate()


if __name__ == '__main__':
    demo1()