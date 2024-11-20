# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:48
# @File    : BenchMarkForJson.py
# @Software: PyCharm

from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
import src.Preference.EnumType as EnumType
import numpy as np
import matplotlib.pyplot as plt


class BenchMark:
    def __init__(self):
        self._fr = Frame(EOP().load())
        self._savedir = "../result/benchmarktest/"
        pass

    def three_body(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
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
            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'planets.npy', np.array(diff))
        diff = np.array(diff)
        ax = diff[:, 0]
        ay = diff[:, 1]
        az = diff[:, 2]
        plt.style.use('grid')
        plt.title('TidePlanets(MERCURY, VENUS, MARS, JUPITER, SATURN)')
        plt.plot(timeList, ax, color='#dd2c00', label='x')
        plt.plot(timeList, ay, color='#00695c', label='y')
        plt.plot(timeList, az, color='#1a237e', label='z')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('acc')
        plt.show()

    def relativistic(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
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
            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'relativistic.npy', np.array(diff))
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

    def aod(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
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
            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.getCS().calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'AODRL06.npy', np.array(diff))
        diff = np.array(diff)
        ax = diff[:, 0]
        ay = diff[:, 1]
        az = diff[:, 2]
        plt.style.use('grid')
        plt.title('AOD1B RL06(2~180)')
        plt.plot(timeList, ax, color='#dd2c00', label='x')
        plt.plot(timeList, ay, color='#00695c', label='y')
        plt.plot(timeList, az, color='#1a237e', label='z')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('acc')
        plt.show()

    def eot11a(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
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
            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(
                self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.getCS().calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'eot11a.npy', np.array(diff))
        diff = np.array(diff)
        ax = diff[:, 0]
        ay = diff[:, 1]
        az = diff[:, 2]
        plt.style.use('grid')
        plt.title('EOT11a ocean tide')
        plt.plot(timeList, ax, color='#dd2c00', label='x')
        plt.plot(timeList, ay, color='#00695c', label='y')
        plt.plot(timeList, az, color='#1a237e', label='z')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('acc')
        plt.show()

    def fes2014b(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
        '''read orbit'''
        with open('../data/Benchmark/satellite/00orbit_itrf.txt') as f:
            contents1 = f.readlines()
            pass
        '''read benchmark'''
        with open('../data/Benchmark/satellite/11oceanTide_fes2014b_with361Admittance_icrf.txt') as f:
            contents2 = f.readlines()
            pass

        diff = []
        timeList = []
        for i in range(5, len(contents1)):
            res1 = np.array(contents1[i].split(), dtype='float')
            time = res1[0] + 19 / 86400
            pos, vel = res1[1:4], res1[4:7]
            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(
                self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.getCS().calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'fes2014.npy', np.array(diff))
        diff = np.array(diff)
        ax = diff[:, 0]
        ay = diff[:, 1]
        az = diff[:, 2]
        plt.style.use('grid')
        plt.title('FES2014b ocean tide')
        plt.plot(timeList, ax, color='#dd2c00', label='x')
        plt.plot(timeList, ay, color='#00695c', label='y')
        plt.plot(timeList, az, color='#1a237e', label='z')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('acc')
        plt.show()

    def atmos_tide(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
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
            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(
                self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.getCS().calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'atmos.npy', np.array(diff))
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

    def gravity_field(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
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
            time = res1[0] + 19 / 86400
            pos, vel = res1[1:4], res1[4:7]
            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(
                self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.getCS().calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'gravity.npy', np.array(diff))
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

    def solid_earth(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
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
            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(
                self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.getCS().calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'solidearth.npy', np.array(diff))
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

    def earth_pole(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
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

            res2 = np.array(contents3[i - 3].split()).astype(np.float64)
            xp, yp = res2[1], res2[2]

            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(
                self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.setEOP(xp, yp).getCS().calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'earthpole.npy', np.array(diff))
        diff = np.array(diff)
        ax = diff[:, 0]
        ay = diff[:, 1]
        az = diff[:, 2]
        plt.style.use('grid')
        plt.title('Earth pole tide')
        plt.plot(timeList, ax, color='#dd2c00', label='x')
        plt.plot(timeList, ay, color='#00695c', label='y')
        plt.plot(timeList, az, color='#1a237e', label='z')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('acc')
        plt.show()

    def ocean_pole(self, FMConfig = ForceModelConfig()):
        assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)
        '''read orbit'''
        with open('../data/Benchmark/satellite/00orbit_itrf.txt') as f:
            contents1 = f.readlines()
            pass
        '''read benchmark'''
        with open('../data/Benchmark/satellite/06oceanPoleTide_icrf.txt') as f:
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

            res2 = np.array(contents3[i - 3].split()).astype(np.float64)
            xp, yp = res2[1], res2[2]

            assem_model = assembleModel.setPosAndVel(pos, vel).setTime(
                self._fr.setTime(time, EnumType.TimeFormat.TAI_MJD))
            acc = assem_model.setEOP(xp, yp).getCS().calculation().getAcceleration()

            ref1 = np.array(contents2[i].split(), dtype='float')
            acc_ref = ref1[1:4]

            diff.append(acc - acc_ref)
            timeList.append(time)
            pass

        np.save(self._savedir + 'oceanpole.npy', np.array(diff))
        diff = np.array(diff)
        ax = diff[:, 0]
        ay = diff[:, 1]
        az = diff[:, 2]
        plt.style.use('grid')
        plt.title('Ocean tide')
        plt.plot(timeList, ax, color='#dd2c00', label='x')
        plt.plot(timeList, ay, color='#00695c', label='y')
        plt.plot(timeList, az, color='#1a237e', label='z')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('acc')
        plt.show()