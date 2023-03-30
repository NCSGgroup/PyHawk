# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:48
# @File    : BenchMarkForJson.py
# @Software: PyCharm

from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.ForceModelConfig import ForceModelConfig
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