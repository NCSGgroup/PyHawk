import sys
sys.path.append('../')
from src.Solver.DesignMatrix import GravityDesignMat
import src.Preference.backups.Pre_Solver as Pre_Solver
import pathlib
import h5py
import src.Preference.EnumType as EnumType
from src.Interface.ArcOperation import ArcData
import numpy as np
import matplotlib.pyplot as plt
from src.Preference.Pre_Accelerometer import AccelerometerConfig
from src.Preference.Pre_Solver import SolverConfig
from src.Preference.Pre_Parameterization import ParameterConfig
import json


def demo1():
    accConfig_A = AccelerometerConfig()
    acc_A = json.load(open('../setting/Calibrate/AccelerometerConfig_A.json', 'r'))
    accConfig_A.__dict__ = acc_A
    accConfig_B = AccelerometerConfig()
    acc_B = json.load(open('../setting/Calibrate/AccelerometerConfig_B.json', 'r'))
    accConfig_B.__dict__ = acc_B

    solverConfig = SolverConfig()
    solverDict = json.load(open('../setting/Calibrate/SolverConfig.json', 'r'))
    solverConfig.__dict__ = solverDict
    dm = GravityDesignMat(0).configure(accConfig=[accConfig_A, accConfig_B], solverConfig=solverConfig, parameterConfig= ParameterConfig)
    # dm.orbit_integration()
    # dm.get_orbit_design_matrix()
    dm.get_sst_design_matrix()


def demo2():
    date_span = ['2008-01-01', '2008-01-31']
    res_dir = pathlib.Path(Pre_Solver.PathOfFiles.DesignMatrixTemp).joinpath(date_span[0] + '_' + date_span[1])
    res_filename = res_dir.joinpath(str(0) + '_AB.hdf5')
    h5 = h5py.File(res_filename, 'r')
    print(h5['r'][:])
    rd = ArcData(date_span=date_span)
    GNV = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    GNVTime = GNV[:, 0]
    GNVR = GNV[:, 1:4]

    diff = h5['r'][:][:, 0:3, 0] - GNVR

    plt.title('2008-1-1 diff', fontsize=24)
    plt.plot(np.array(GNVTime), diff[:, 0], label="X")
    plt.plot(np.array(GNVTime), diff[:, 1], label="Y")
    plt.plot(np.array(GNVTime), diff[:, 2], label="Z")
    plt.xlabel('GPS time [sec]', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    # plt.grid(ls='--')
    plt.show()


if __name__ == '__main__':
    demo1()
    # demo2()