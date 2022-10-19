from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat
from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.ConfigJson.ConfigByJson import ConfigFMbyJson
from src.ODE.GravimetryODE import GravimetryODE
import json
import numpy as np
import src.Preference.EnumType as EnumType
import src.Preference.Pre_ForceModel as Pre_ForceModel


def demo1():
    with open('../setting/BenchMarkTest_ForceModel.setting.json') as load_f:
        configA = json.load(load_f)
    with open('../setting/GraceTestForceModel_B.json') as load_f:
        configB = json.load(load_f)
    fr = Frame(EOP(True).load())

    '''sat state'''
    posA = np.array([-4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06])
    velA = np.array([3000, 5000, 6000])
    posB = np.array([-4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06])
    velB = np.array([3000, 5000, 6000])
    x0 = 504964860

    '''sat force'''
    fr.setTime(504964830, TimeFormat.GPS_second)
    for force in EnumType.GravHarType:
        print(1, force, Pre_ForceModel.instance_GravHar[force])
    pc = ConfigFMbyJson().configure(configA)
    for force in EnumType.GravHarType:
        print(2, force, Pre_ForceModel.instance_GravHar[force])
    # pc = ConfigFMbyJson().configure(configB)
    # for force in EnumType.ForceType:
    #     print(3, force, Pre_ForceModel.instance_GravHar[force])
    # modelA = Assemble2ndDerivative()
    # print(modelA.ForceType)
    # modelA = modelA.setPosAndVel(posA, velA).setTime(fr)
    # pc = ConfigFMbyJson().configure(configA)
    # modelB = Assemble2ndDerivative()
    # modelB = modelB.setPosAndVel(posA, velA).setTime(fr)
    # print(modelB.ForceType)
    # print(modelA.getAcceleration())
    # print(modelB.getAcceleration())


if __name__ == '__main__':
    demo1()