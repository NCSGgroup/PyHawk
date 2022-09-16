from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat, GravHarType
from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.ForceModel.MainPropagator import ConfigFMbyJson, Propagator
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
    x0 = 504964860

    '''sat force'''
    fr.setTime(504964830, TimeFormat.GPS_second)
    pc = ConfigFMbyJson().configure(configA)
    modelA = Assemble2ndDerivative()
    # modelA = modelA.setPosAndVel(posA, velA).setTime(fr)
    pc = ConfigFMbyJson().configure(configA)
    modelB = Assemble2ndDerivative()
    # modelB = modelB.setPosAndVel(posA, velA).setTime(fr)
    # print(modelA.getAcceleration())
    # print(modelB.getAcceleration())

    '''ode'''
    propagator = Propagator(fr).setInitial(x0, posA, velA).setForceModel(modelA)
    X_GJ, R_GJ, v = propagator.propagate()
    print(X_GJ)
    print(R_GJ)
    print(v)


if __name__ == '__main__':
    demo1()