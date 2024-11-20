from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import src.Preference.EnumType as EnumType
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
import numpy as np
import matplotlib.pyplot as plt

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
    with open('../data/Benchmark/satellite/08aod1b_RL06_icrf.txt') as f:
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

    np.save(savedir + 'AODRL06.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('dealiasing')
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


def demo_solidEarth_tide():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_itrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/04solidEarthTide_icrf.txt') as f:
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

    np.save(savedir + 'solidearth.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('Solid tide')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


def demo_relativistic():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_icrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/07relativistic_icrf.txt') as f:
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

    np.save(savedir + 'relativistic.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('Relativistic')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


def demo_earth_pole_tide():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_itrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/05poleTide_icrf.txt') as f:
        contents2 = f.readlines()
        pass
    '''read EOP'''
    with open('../data/Benchmark/satellite/01earthRotation_interpolatedEOP.txt') as f:
        contents3 = f.readlines()
        pass

    diff = []
    timeList = []
    for i in range(5, len(contents1)):
        res1 = np.array(contents1[i].split(), dtype='float')
        time = res1[0] + 19 / 86400
        pos, vel = res1[1:4], res1[4:7]
        assem_model = assem_model.setPosAndVel(pos, vel).setTime(fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
        acc = assem_model.getAcceleration()

        res2 = np.array(contents3[i - 3].split()).astype(np.float64)
        xp, yp = res2[1], res2[2]

        ref1 = np.array(contents2[i].split(), dtype='float')
        acc_ref = ref1[1:4]

        diff.append(acc - acc_ref)
        timeList.append(time)
        pass

    np.save(savedir + 'earthpole.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('Earth pole')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


def demo_atm_tide():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_itrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/09aod1b_atmosphericTides_icrf.txt') as f:
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

    np.save(savedir + 'atm.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('Atmos tide')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


def demo_direct_tide_Moon():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_icrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/03directTideMoon_icrf.txt') as f:
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

    np.save(savedir + 'moon.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('Moon tide')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


def demo_direct_tide_Sun():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_icrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/03directTideSun_icrf.txt') as f:
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

    np.save(savedir + 'sun.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('Sun tide')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


def demo_3rd_body():
    fr = Frame(EOP().load())
    assem_model = Assemble2ndDerivative()

    '''read orbit'''
    with open('../data/Benchmark/satellite/00orbit_icrf.txt') as f:
        contents1 = f.readlines()
        pass
    '''read benchmark'''
    with open('../data/Benchmark/satellite/03directTidePlanets_icrf.txt') as f:
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

    np.save(savedir + 'planets.npy', np.array(diff))
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    plt.title('Plants tide')
    plt.plot(timeList, ax, color='#dd2c00', label='x')
    plt.plot(timeList, ay, color='#00695c', label='y')
    plt.plot(timeList, az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD')
    plt.ylabel('acc')
    plt.show()


if __name__ == '__main__':
    # demo_gravity_field()
    # demo_aod()
    # demo_ocean_tide()
    demo_solidEarth_tide()
    # demo_relativistic()
    # demo_earth_pole_tide()
    # demo_atm_tide()
    # demo_direct_tide_Moon()
    # demo_direct_tide_Sun()
    # demo_3rd_body()