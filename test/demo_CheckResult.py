# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2024/1/4 16:05
# @File    : demo_CheckResult.py
# @Software: PyCharm
import sys
sys.path.append("../")
import src.Preference.Pre_AdjustOrbit as Pre_AdjustOrbit
from src.Interface.ArcOperation import ArcData
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.EnumType import Payload
import matplotlib.pyplot as plt
import pathlib
import json
import h5py
import numpy as np


def readDate():
    interface_config = InterfaceConfig()
    interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict
    rd = ArcData(interfaceConfig=interface_config)

    date_span = ["2009-01-01", "2009-01-31"]
    res_dir = "../temp/StateVector/" + str(date_span[0]) + '_' + str(date_span[1])

    for i in range(0, 1):
        gnv = rd.getData(arc=i, kind=Payload.GNV, sat='A')
        StateName = res_dir + '/' + str(i) + '_AB.hdf5'
        StateH5 = h5py.File(StateName, "r")
        StateGNV = StateH5['r'][:]
        diff = (StateGNV[:, 0:3, 0] - gnv[:, 1:4]) * 1000
        plot(gnv[:, 0], diff[:, 0], diff[:, 1], diff[:, 2])

        print(__rms(diff[:, 0]))
        print(__rms(diff[:, 1]))
        print(__rms(diff[:, 2]))
    # StateVector = "../temp/StateVector/2010-11-01_2010-11-30/51_AB.hdf5"
    # StateVectorh5 = h5py.File(StateVector, "r")
    # Stategnv = StateVectorh5['r'][:]
    #
    # RawData = "../temp/RawData/2010-11-01_2010-11-30.hdf5"
    # RawDatah5 = h5py.File(RawData, "r")
    # for i in range(0, 3):
    #     acc = RawDatah5['ACC_A_arc' + str(i)][:]
    #     sca = RawDatah5['SCA_A_arc' + str(i)][:]
    #     gnv = RawDatah5['GNV_A_arc'+str(i)][:][:, :4]
    #     kin = RawDatah5['KinematicOrbit_A_arc'+str(i)][:]
    #     plot(gnv[:, 0], gnv[:, 1], gnv[:, 2], gnv[:, 3])
    # StateVectorh5.close()
    # RawDatah5.close()


def plot(t, x, y, z):
    t = np.arange(len(x))
    plt.figure(figsize=(12, 6))
    plt.subplot(311)
    plt.title('{}-{}-{}-{}'.format('GNV', 'A', '2009-01', 'arc4'), fontsize=24)
    plt.scatter(t, x, marker='o')
    plt.yticks(fontsize=18)
    plt.ylabel(r'$X/mm$', fontsize=20)

    plt.subplot(312)
    plt.scatter(t, y, marker='o')
    plt.yticks(fontsize=18)
    plt.ylabel(r'$Y/mm$', fontsize=20)

    plt.subplot(313)
    plt.scatter(t, z, marker='o')
    plt.xlabel('GPS Time', fontsize=20)
    plt.ylabel(r'$Z/mm$', fontsize=20)
    plt.yticks(fontsize=18)

    # plt.grid(ls='--')
    plt.show()


def __rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


if __name__ == '__main__':
    readDate()
    # filename = '../temp/RawData/2006-03-01_2006-03-31.hdf5'
    # h5 = h5py.File(filename, 'r')
    # h5.keys()