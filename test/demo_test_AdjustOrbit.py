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


def demo1():
    date_span = ['2008-01-01', '2008-01-31']
    res_dir = pathlib.Path('../temp/OrbitAdjustment').joinpath(date_span[0] + '_' + date_span[1])
    res_filename = res_dir.joinpath(str(0) + '_AB.hdf5')
    h5 = h5py.File(res_filename, 'r')
    print(h5.keys())
    print(h5['TransitionMatrix'][:])
    print(h5['Accelerometer'][:])


def analysis():
    date_span = ['2008-01-01', '2008-01-31']
    res_dir = pathlib.Path('../temp/DesignMatrixSH').joinpath(date_span[0] + '_' + date_span[1])

    interface_config = InterfaceConfig()
    interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict
    rd = ArcData(interfaceConfig=interface_config)

    RMSX = []
    RMSY = []
    RMSZ = []
    for i in range(0, 240):
        res_filename = res_dir.joinpath(str(i) + '_AB.hdf5')
        h5 = h5py.File(res_filename, 'r')
        rAB = h5['r'][:]
        rA = rAB[:, 3:6, 0]
        gnv = rd.getData(arc=i, kind=Payload.GNV, sat='B')[:, 1:4]

        if np.shape(gnv) == np.shape(rA):
            diff = gnv - rA
            RMSX.append(rms(diff[:, 0]))
            RMSY.append(rms(diff[:, 1]))
            RMSZ.append(rms(diff[:, 2]))

    np.save('OrbitRMSX.npy', RMSX)
    np.save('OrbitRMSY.npy', RMSY)
    np.save('OrbitRMSZ.npy', RMSZ)
    pass


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


if __name__ == '__main__':
    # demo1()
    # analysis()
    X = np.load('OrbitRMSX.npy')
    Y = np.load('OrbitRMSY.npy')
    Z = np.load('OrbitRMSZ.npy')
    plt.subplot(311)
    plt.title('GRACE-B(2008-1-1~2008-1-31)', fontsize=24)
    plt.plot(np.arange(len(X)), X * 1000, marker='o', label="Direction X")
    plt.yticks(fontsize=18)
    plt.ylabel(r'$X/mm$', fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.subplot(312)
    plt.plot(np.arange(len(Y)), Y * 1000, marker='o', label="Direction Y")
    plt.yticks(fontsize=18)
    plt.ylabel(r'$Y/mm$', fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.subplot(313)
    plt.plot(np.arange(len(Z)), Z * 1000, marker='o', label="Direction Z")
    plt.xlabel('arcNo', fontsize=20)
    plt.ylabel(r'$Z/mm$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    # plt.grid(ls='--')
    plt.show()