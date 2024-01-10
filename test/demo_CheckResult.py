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
    filename = "../temp/RawData/2009-03-01_2009-03-31.hdf5"
    h5 = h5py.File(filename, "r")
    for i in range(71, 72):

        gnv = h5['GNV_A_arc'+str(i)][:][:, :4]
        kin = h5['KinematicOrbit_A_arc'+str(i)][:]
        plot(gnv[:, 0], gnv[:, 1], gnv[:, 2], gnv[:, 3])
    h5.close()


def plot(t, x, y, z):
    t = (t - t[0]) / 5
    plt.figure(figsize=(12, 6))
    plt.subplot(311)
    plt.title('{}Orbit-{}-{}-{}'.format('GNV', 'A', '2009-03', 'arc69'), fontsize=24)
    plt.plot(t, x, marker='o')
    plt.yticks(fontsize=18)
    plt.ylabel(r'$X/mm$', fontsize=20)

    plt.subplot(312)
    plt.plot(t, y, marker='o')
    plt.yticks(fontsize=18)
    plt.ylabel(r'$Y/mm$', fontsize=20)

    plt.subplot(313)
    plt.plot(t, z, marker='o')
    plt.xlabel('arcNo', fontsize=20)
    plt.ylabel(r'$Z/mm$', fontsize=20)
    plt.yticks(fontsize=18)

    plt.grid(ls='--')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    readDate()
