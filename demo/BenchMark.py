from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import src.Preference.EnumType as EnumType
from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.ODE.GravimetryODE import GravimetryODE
from src.Interface.ArcOperation import ArcData
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

savedir = "../result/benchmarktest/"


def demo_gravity_field():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_itrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/02gravityfield_itrf.txt') as f:
        contents2 = f.readlines()
        pass

    diff = []
    timeList = []
    for i in range(5, len(contents1)):
        res1 = np.array(contents1[i].split(), dtype='float')
        time = res1[0]
        pos, vel = res1[1:4], res1[4:7]
        assem_model = assem_model.setPosAndVel(pos, vel).setTime(fr.setTime(time))
        acc = assem_model.getAcceleration()

        ref1 = np.array(contents2[i].split(), dtype='float')
        acc_ref = ref1[1:4]

        diff.append(acc - acc_ref)
        timeList.append(time)
        pass

    np.save(savedir + 'gravity.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('GravityField')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


def demo_aod():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_itrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/02gravityfield_itrf.txt') as f:
        contents2 = f.readlines()
        pass

    diff = []
    timeList = []
    for i in range(5, len(contents1)):
        res1 = np.array(contents1[i].split(), dtype='float')
        time = res1[0]
        pos, vel = res1[1:4], res1[4:7]
        assem_model = assem_model.setPosAndVel(pos, vel).setTime(fr.setTime(time))
        acc = assem_model.getAcceleration()

        ref1 = np.array(contents2[i].split(), dtype='float')
        acc_ref = ref1[1:4]

        diff.append(acc - acc_ref)
        timeList.append(time)
        pass

    np.save(savedir + 'gravity.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('GravityField')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


def demo_ocean_tide():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_itrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/10oceanTide_eot11a_with256Admittance_icrf.txt') as f:
        contents2 = f.readlines()
        pass

    diff = []
    timeList = []
    for i in range(5, len(contents1)):
        res1 = np.array(contents1[i].split(), dtype='float')
        time = res1[0] + 19 / 86400
        pos, vel = res1[1:4], res1[4:7]
        assem_model = assem_model.setPosAndVel(pos, vel).setTime(fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
        acc = assem_model.getAcceleration()

        ref1 = np.array(contents2[i].split(), dtype='float')
        acc_ref = ref1[1:4]

        diff.append(acc - acc_ref)
        timeList.append(time)
        pass

    np.save(savedir + 'eot11a.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('EOT11a')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


if __name__ == '__main__':
    demo_gravity_field()
    # demo_ocean_tide()
