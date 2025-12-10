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
import glob
import os
from pathlib import Path
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.Pre_Frame import FrameConfig
from src.Auxilary.GeoMathKit import GeoMathKit
from scipy.linalg import toeplitz, cholesky, eigh, solve_triangular


def readDate():

    IAG_kbrr = []
    IAG_time = []

    folder = Path("../temp/kbr/")
    for f in folder.glob("*.txt"):
        data = np.loadtxt(f)  # 假设是纯数值表格
        IAG_time.extend(list(data[:, 0]))
        IAG_kbrr.extend(list(data[:, 4]))

    pre_RMS = []
    pre_time = []
    res_dir = pathlib.Path().joinpath("../temp/RangeRate", "2019-01-01_2019-01-31")
    for i in np.arange(0, 124):
        res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')

        if os.path.exists(res_filename):
            h5 = h5py.File(res_filename, 'r')
            if len(h5.keys()) != 0:
                postfit = h5['post_residual'][()]
                pre_time.extend(list(h5['post_t'][()]))
                pre_RMS.extend(list(postfit))

            h5.close()
    IAG_time = IAG_time[:10000]
    pre_time = pre_time[:10000]
    pre_RMS = pre_RMS[:10000]
    IAG_kbrr = IAG_kbrr[:10000]
    commonTime = set(np.array(IAG_time).astype(np.int64))
    commonTime = commonTime & set(np.array(pre_time).astype(np.int64))
    commonTime = np.array(list(commonTime))
    commonTime.sort()

    new_index1 = [IAG_time.index(x) for x in commonTime]
    new_index2 = [pre_time.index(x) for x in commonTime]
    pre_RMS = np.array(pre_RMS)[new_index2]
    IAG_kbrr = np.array(IAG_kbrr)[new_index1]

    # plt.figure(dpi=300, figsize=(16, 8))
    plt.title('KBRR-RMS-{}'.format("2019-01"), fontsize=24)
    plt.plot(np.arange(len(pre_RMS)), [i * 1000000 for i in pre_RMS], marker='o', color='green', label="pyhawk kbrr")
    plt.plot(np.arange(len(IAG_kbrr)), [i * 1000000 for i in IAG_kbrr], marker='*', color='red', label="IGG kbrr")
    plt.yticks(fontsize=18)
    # plt.ylim(0, 0.7)
    plt.ylabel(r'$X/um$', fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.show()

    # interface_config = InterfaceConfig()
    # interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    # interface_config.__dict__ = interfaceDict
    # rd = ArcData(interfaceConfig=interface_config)
    #
    # date_span = ["2009-01-01", "2009-01-31"]
    # res_dir = "../temp/StateVector/" + str(date_span[0]) + '_' + str(date_span[1])
    #
    # for i in range(0, 1):
    #     gnv = rd.getData(arc=i, kind=Payload.GNV, sat='A')
    #     kbrr = rd.getData(arc=i, kind=Payload.KBR, sat='A')
    #
    #     StateName = res_dir + '/' + str(i) + '_AB.hdf5'
    #     StateH5 = h5py.File(StateName, "r")
    #     StateGNV = StateH5['r'][:]
    #     diff = (StateGNV[:, 0:3, 0] - gnv[:, 1:4])
    #     # diff = (StateGNV[:, 0:3, 0] - StateGNV[:, 3:6, 0]) * 0.001
    #     plot(gnv[:, 0], diff[:, 0], diff[:, 1], diff[:, 2], arc=i)
    #
    #     print(__rms(diff[:, 0]))
    #     print(__rms(diff[:, 1]))
    #     print(__rms(diff[:, 2]))
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

def xyzspace():
    datespan = ['2019-01-01', '2019-01-31']
    frame_config = FrameConfig()
    frame = json.load(open(('../setting/Calibrate/FrameConfig.json'), 'r'))
    frame_config.__dict__ = frame

    eop = EOP().configure(frameConfig=frame_config).load()
    fr = Frame(eop).configure(frameConfig=frame_config)

    raw_data_path = "../temp/RawData"
    filename = pathlib.Path().joinpath(raw_data_path, datespan[0] + '_' + datespan[1] + ".hdf5")

    IAG_kbrr = []
    IAG_time = []

    folder = Path("../temp/kbr/")
    for f in folder.glob("*.txt"):
        data = np.loadtxt(f)  # 假设是纯数值表格
        IAG_time.extend(list(data[:, 0]))
        IAG_kbrr.extend(list(data[:, 4]))

    h5data = h5py.File(filename, "r")

    Lon = []
    Lat = []
    Prefit = []
    for j in np.arange(0, 124):
        GNV = h5data['GNV_A_arc' + str(j)][()]
        GNVTime = GNV[:, 0]
        GNVR = GNV[:, 1:4]

        commonTime = set(np.array(IAG_time).astype(np.int64))
        commonTime = commonTime & set(GNVTime.astype(np.int64))
        commonTime = np.array(list(commonTime))
        commonTime.sort()

        new_index1 = [IAG_time.index(x) for x in commonTime]
        new_index2 = [list(GNVTime).index(x) for x in commonTime]

        newTime = GNVTime[new_index2]
        newGNVR = GNVR[new_index2]

        Prefit.extend(list(np.array(IAG_kbrr)[new_index1]))

        for k in range(len(newTime)):
            time = newTime[k]
            pos = newGNVR[k]
            fr = fr.setTime(float(time))
            itrs = fr.PosGCRS2ITRS(np.array(pos).astype(float), fr.getRotationMatrix)
            lon, lat, r = GeoMathKit.CalcPolarAngles(itrs)
            Lon.append(lon)
            Lat.append(lat)

    Lon = np.degrees(np.array(Lon))
    Lat = np.degrees(np.array(Lat))
    Prefit = np.array(Prefit)

    pre_filename = '../result/IGG-' + datespan[0] + '_' + datespan[1] + '.hdf5'
    pre_h5 = h5py.File(pre_filename, "w")
    pre_h5.create_dataset('lon', data=Lon)
    pre_h5.create_dataset('lat', data=Lat)
    pre_h5.create_dataset('prefit', data=Prefit)
    pre_h5.close()

    # interface_config = InterfaceConfig()
    # interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
    # interface_config.__dict__ = interfaceDict
    # rd = ArcData(interfaceConfig=interface_config)
    #
    # date_span = ["2009-01-01", "2009-01-31"]
    # res_dir = "../temp/StateVector/" + str(date_span[0]) + '_' + str(date_span[1])
    #
    # for i in range(0, 1):
    #     gnv = rd.getData(arc=i, kind=Payload.GNV, sat='A')
    #     kbrr = rd.getData(arc=i, kind=Payload.KBR, sat='A')
    #
    #     StateName = res_dir + '/' + str(i) + '_AB.hdf5'
    #     StateH5 = h5py.File(StateName, "r")
    #     StateGNV = StateH5['r'][:]
    #     diff = (StateGNV[:, 0:3, 0] - gnv[:, 1:4])
    #     # diff = (StateGNV[:, 0:3, 0] - StateGNV[:, 3:6, 0]) * 0.001
    #     plot(gnv[:, 0], diff[:, 0], diff[:, 1], diff[:, 2], arc=i)
    #
    #     print(__rms(diff[:, 0]))
    #     print(__rms(diff[:, 1]))
    #     print(__rms(diff[:, 2]))
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


def plot(t, x, y, z, arc):
    t = np.arange(len(x))
    plt.figure(figsize=(12, 6))
    plt.subplot(311)
    plt.title('{}-{}-{}-{}'.format('GNV', 'A', '2009-01', arc), fontsize=24)
    plt.scatter(t, x, marker='o')
    plt.yticks(fontsize=18)
    plt.ylabel(r'$X/km$', fontsize=20)

    plt.subplot(312)
    plt.scatter(t, y, marker='o')
    plt.yticks(fontsize=18)
    plt.ylabel(r'$Y/km$', fontsize=20)

    plt.subplot(313)
    plt.scatter(t, z, marker='o')
    plt.xlabel('GPS Time', fontsize=20)
    plt.ylabel(r'$Z/km$', fontsize=20)
    plt.yticks(fontsize=18)

    # plt.grid(ls='--')
    plt.savefig('../result/checkorbit/diff-gnv-A-{}.png'.format(arc))
    # plt.show()

def postValue():
    postKBRR = "../temp/PostKBRR/2019-01-01-2019-01-31"
    postOrbit = "../temp/PostOrbit/2019-01-01-2019-01-31"
    postPar = "../temp/PostParameter/2019-01-01-2019-01-31"

    for i in range(0, 123):
        postKBRRname = postKBRR + "/RangeRate_" + str(i) + ".hdf5"
        postOrbitname_a = postOrbit + "/Orbit_" + str(i) + "A.hdf5"
        postOrbitname_b = postOrbit + "/Orbit_" + str(i) + "B.hdf5"

        postKBRRh5 = h5py.File(postKBRRname, "r")
        postOrbith5_a = h5py.File(postOrbitname_a, "r")
        postOrbith5_b = h5py.File(postOrbitname_b, "r")

        res_kbrr = postKBRRh5['post_kbrr'][:]
        res_orbit_a = postOrbith5_a['post_Orbit'][:]
        res_orbit_b = postOrbith5_b['post_Orbit'][:]

        n = len(res_kbrr)
        acov_kbrr = estimate_acov(res_kbrr, max_lag=(int(np.sqrt(n))))
        acov_orbit_a = estimate_acov(res_orbit_a)
        acov_orbit_b = estimate_acov(res_orbit_b)
        '''构建协方差矩阵'''
        first_col = np.zeros(n)
        first_col[:len(acov_kbrr)] = acov_kbrr
        Sigma = toeplitz(first_col)
        '''正定化'''
        diag_scale = np.max(np.abs(np.diag(Sigma)))
        if diag_scale <= 0:
            diag_scale = 1.0
        Sigma += np.eye(n) * (1e-8 * diag_scale)
        Sigma = nearest_pos_def(Sigma, eps_rel=1e-8)
        '''Cholesky白化'''
        L = cholesky(Sigma, lower=True)

        # # A is n x p
        # Aw = solve_triangular(L, A, lower=True)
        # lw = solve_triangular(L, l, lower=True)
        #
        # # GLS via OLS on whitened system
        # x_gls, *_ = np.linalg.lstsq(Aw, lw, rcond=None)
        print(1)
    pass


def estimate_acov(residuals, max_lag=50):
    """
    用残差估计自协函数（ACF），无偏估计版本。
    """
    residuals = np.asarray(residuals)
    n = len(residuals)
    mean = residuals.mean()
    r = residuals - mean

    acov = []
    for k in range(max_lag + 1):
        cov = np.sum(r[:n-k] * r[k:]) / (n - k)
        acov.append(cov)
    return np.array(acov)


def nearest_pos_def(mat, eps_rel=1e-8):
    mat = 0.5 * (mat + mat.T)
    vals, vecs = eigh(mat)
    max_val = vals[-1] if vals.size>0 else 0.0
    tol = eps_rel * max(1.0, max_val)
    vals_clipped = np.clip(vals, tol, None)
    mat_pd = (vecs @ np.diag(vals_clipped)) @ vecs.T
    return (mat_pd + mat_pd.T) / 2.0


def __rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


if __name__ == '__main__':
    # xyzspace()
    postValue()
    # filename = '../temp/RawData/2006-03-01_2006-03-31.hdf5'
    # h5 = h5py.File(filename, 'r')
    # h5.keys()