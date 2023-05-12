import sys
sys.path.append("../")
from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType
import numpy as np
from tqdm import tqdm
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.Solver.AdjustOrbit import AdjustOrbit
import matplotlib.pyplot as plt
import src.Preference.backups.Pre_AdjustOrbit as Pre_AdjustOrbit
import pathlib
import h5py
import src.Preference.backups.Pre_Solver as Pre_Solver
import multiprocessing
from src.Solver.DesignMatrix import GravityDesignMat
from src.Solver.NEQ import GravityNEQperArc
import src.Preference.backups.Pre_NEQ as Pre_NEQ
import src.Preference.backups.Pre_Interface as Pre_Interface
from src.Auxilary.SortCS import SortCS
from src.Auxilary.GeoMathKit import GeoMathKit
import src.Preference.backups.Pre_Parameterization as Pre_Parameterization
from src.Interface.LoadSH import LoadGif48
from src.Preference.backups.Pre_ForceModel import PathOfFiles
from src.Preference.Pre_Accelerometer import AccelerometerConfig
import json


def adjust():
    # """step 1: Read raw data"""
    # fr = Frame(EOP().load())
    # s = GRACE_OLD_RL02().configure().setDate()
    # s.read_double_sat()
    # """step 2: Convert orbit from ITRS frame into GCRS frame"""
    # kine = KinematicOrbit(L1b=s).configure()
    # # kine.read_double_sat()
    # kine.GNVandKiOrbitItrs2Gcrs(fr)
    # """step 3: Gap fix"""
    # gf = GapFix(s).configure()
    # gf.fix_all().makeReport()
    # arc = ArcSelect(gf).configure()
    # arc.unpackArcs().makeReport()

    p1 = multiprocessing.Process(target=demo1, args=(0, 41))
    p2 = multiprocessing.Process(target=demo1, args=(41, 82))
    p3 = multiprocessing.Process(target=demo1, args=(82, 123))
    p4 = multiprocessing.Process(target=demo1, args=(123, 164))
    p5 = multiprocessing.Process(target=demo1, args=(164, 205))
    p6 = multiprocessing.Process(target=demo1, args=(205, 243))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()


def demo():

    """step 1: Read raw data"""
    # fr = Frame(EOP().load())
    # s = GRACE_NEW_RL02().configure().setDate()
    # s.read_double_sat()
    # """step 2: Convert orbit from ITRS frame into GCRS frame"""
    # kine = KinematicOrbit(L1b=s).configure()
    # kine.read_double_sat()
    # kine.GNVandKiOrbitItrs2Gcrs(fr)
    # """step 3: Gap fix"""
    # gf = GapFix(s).configure()
    # gf.fix_all().makeReport()
    # arc = ArcSelect(gf).configure()
    # arc.unpackArcs().makeReport()
    """step 5: init AdjustOrbit"""
    date_span = ['2008-01-01', '2008-01-31']
    accConfig_A = AccelerometerConfig()
    acc_A = json.load(open('../setting/Calibrate/AccelerometerConfig_A.json', 'r'))
    accConfig_A.__dict__ = acc_A
    accConfig_B = AccelerometerConfig()
    acc_B = json.load(open('../setting/Calibrate/AccelerometerConfig_B.json', 'r'))
    accConfig_B.__dict__ = acc_B
    AdjustOrbit(accConfig=[accConfig_A, accConfig_B],date_span=date_span).\
        configure(Orbit_var_2nd_diff, arcNo=0).calibrate(iteration=2)
    # initTime, initR, endTime, endR = adjust.draw()
    # firstR, secondR, thirdR = adjust.getR()
    # np.savez('adjust.npz', R1=firstR, R2=secondR, R3=thirdR)

    # RR = RangeRate(23)
    # RR.configure()
    # RR.calibrate()
    # '''step 6: get DesignMatrix'''
    # dm = GravityDesignMat(0).configure(accConfig=[accConfig_A, accConfig_B])
    # dm.orbit_integration()
    # dm.get_orbit_design_matrix()
    # dm.get_sst_design_matrix()
    # # '''setp 8: get NEQ'''
    # neq = GravityNEQperArc(0)
    # neq.configure().get_neq()


def demo1(start, end):
    date_span = ['2008-01-01', '2008-01-31']
    accConfig_A = AccelerometerConfig()
    acc_A = json.load(open('../setting/Calibrate/AccelerometerConfig_A.json', 'r'))
    accConfig_A.__dict__ = acc_A
    accConfig_B = AccelerometerConfig()
    acc_B = json.load(open('../setting/Calibrate/AccelerometerConfig_B.json', 'r'))
    accConfig_B.__dict__ = acc_B

    # """step 5: init AdjustOrbit"""
    # for i in range(start, end):
    #     AdjustOrbit(accConfig=[accConfig_A, accConfig_B]).configure(Orbit_var_2nd_diff, arcNo=i).calibrate(iteration=2)
    # """step 6: adjustSST"""
    # for i in range(start, end):
    #     RR = RangeRate(i, date_span=date_span)
    #     RR.configure()
    #     RR.calibrate()
    """step 7: DesignMatrix"""
    for i in range(start, end):
        dm = GravityDesignMat(i).configure(accConfig=[accConfig_A, accConfig_B])
        dm.orbit_integration()
        dm.get_orbit_design_matrix()
        dm.get_sst_design_matrix()
    '''setp 8: get NEQ'''
    for i in range(start, end):
        neq = GravityNEQperArc(i)
        neq.configure().get_neq()


def getRMS():
    dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp). \
        joinpath('2008-02-01_2008-02-29')
    RMSX = []
    RMSY = []
    RMSZ = []
    for i in tqdm(range(0, 231), desc='solve Adjust RMS: '):
        filename = dir.joinpath(str(i) + '_AB.hdf5')
        h5 = h5py.File(filename, 'r')
        orbit_nominal = h5['r'][()][:, 0:3, 0]
        h5.close()

        rd = ArcData()
        GNV = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
        GNVTime = GNV[:, 0]
        GNVR = GNV[:, 1:4]

        diff = orbit_nominal - GNVR
        RMSX.append(rms(diff[:, 0]))
        RMSY.append(rms(diff[:, 1]))
        RMSZ.append(rms(diff[:, 2]))
    np.savez('adjustRMS.npz', RMSX=RMSX, RMSY=RMSY, RMSZ=RMSZ)


def getKBRRRMS():
    dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.RangeRateTemp). \
        joinpath('2008-01-01_2008-01-31')
    RMS = []

    for i in tqdm(range(0, 243), desc='solve Adjust RMS: '):
        res_filename = dir.joinpath('RangeRate_' + str(i) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')

        postfit = h5['post_residual'][()]
        RMS.append(rms(postfit))
        h5.close()
    np.savez('2008-1-KBRRRMS.npz', RMS = RMS)


def drawRMS():
    R = np.load("adjustRMS.npz")
    RMSX = R['RMSX']
    RMSY = R['RMSY']
    RMSZ = R['RMSZ']

    listX = []
    listY = []
    listZ = []
    for i in range(0, 29):
        listX.append(sum(RMSX[i * 8:(i+1) * 8]) / 8)
        listY.append(sum(RMSY[i * 8:(i+1) * 8]) / 8)
        listZ.append(sum(RMSZ[i * 8:(i+1) * 8]) / 8)

    plt.style.use(['grid'])
    plt.title('AdjustRMS(2008-2-1~2008-2-29)', fontsize=24)
    plt.plot(np.arange(len(RMSX)), RMSX, linestyle='--', marker='o', label="RMS-X")
    plt.plot(np.arange(len(RMSY)), RMSY, linestyle='--', marker='s', label="RMS-Y")
    plt.plot(np.arange(len(RMSZ)), RMSZ, linestyle='--', marker='^', label="RMS-Z")
    plt.xlabel('arcNo', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    # plt.grid(ls='--')
    plt.show()


def draw():
    dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp). \
        joinpath('2008-01-01_2008-01-31')
    filename = dir.joinpath(str(0) + '_AB.hdf5')
    h5 = h5py.File(filename, 'r')
    orbit_nominal = h5['r'][()][:, 0:3, :]
    print(np.shape(orbit_nominal))
    h5.close()

    # filename2 = dir.joinpath(str(0) + '_A.hdf5')
    # h52 = h5py.File(filename2, 'r')
    # orbit_nominal2 = h52['r'][()]
    # print(np.shape(orbit_nominal2))
    # h5.close()
    rd = ArcData()
    GNV = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    GNVTime = GNV[:, 0]
    GNVR = GNV[:, 1:4]

    diff = orbit_nominal[:, :, 0] - GNVR

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


def drawMatrix():
    rd = ArcData()
    GNV = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    GNVTime = GNV[:, 0]
    GNVR = GNV[:, 1:4]

    dir = pathlib.Path(Pre_Solver.PathOfFiles.DesignMatrixTemp). \
        joinpath('2016-01-01_2016-01-31')
    filename = dir.joinpath('Orbit_0_AB.hdf5')
    h5 = h5py.File(filename, 'r')
    global_r = h5['local_r'][()][:, 0:3, :]
    # global_v = h5['global_v'][()]
    # local_r = h5['local_r'][()][:,0:3,:]
    # local_v = h5['global_r'][()]
    h5.close()
    filename2 = dir.joinpath('Orbit_0_A.hdf5')
    h5 = h5py.File(filename2, 'r')
    # print(h5.keys())
    global_r1 = h5['local_r'][()]
    # global_v = h5['global_v'][()]
    # local_r1 = h5['local_r'][()]
    # local_v = h5['global_r'][()]
    h5.close()

    # diff = global_r[:, 0, :] - global_r1[:, 0, :]
    diff = global_r - global_r1

    plt.title('2008-1-1 diff', fontsize=24)
    plt.plot(np.array(GNVTime), diff[:, 0, 0], label="X")
    plt.plot(np.array(GNVTime), diff[:, 0, 1], label="Y")
    plt.plot(np.array(GNVTime), diff[:, 0, 2], label="Z")
    plt.xlabel('GPS time [sec]', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    # plt.grid(ls='--')
    plt.show()


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


def cacluateNL():
    res_dir = pathlib.Path(Pre_NEQ.PathOfFiles.NormalEqTemp).joinpath\
        (Pre_Interface.date_span[0] + '_' + Pre_Interface.date_span[1])
    # res_filename = res_dir.joinpath(str(0) + '.hdf5')
    # h5 = h5py.File(res_filename, 'r')
    NMatrix = np.zeros((3717, 3717))
    LMatrix = np.zeros((3717, 1))
    for i in tqdm(range(0, 243), desc='Solve normal equations: '):
        res_filename = res_dir.joinpath(str(i) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        # NMatrix = NMatrix + h5['Orbit_N_A'][:] + h5['Orbit_N_B'][:]
        # LMatrix = LMatrix + h5['Orbit_l_A'][:] + h5['Orbit_l_B'][:]
        # NMatrix = NMatrix + h5['RangeRate_N'][:]
        # LMatrix = LMatrix + h5['RangeRate_l'][:].reshape((3717, 1))
        NMatrix = NMatrix + h5['Orbit_N_A'][:] + h5['Orbit_N_B'][:] + h5['RangeRate_N'][:] * 2.5E11
        LMatrix = LMatrix + h5['Orbit_l_A'][:] + h5['Orbit_l_B'][:] + h5['RangeRate_l'][:].reshape((3717, 1)) * 2.5E11
    x = np.linalg.lstsq(NMatrix, LMatrix[:, 0], rcond=None)[0]
    print(x)
    np.savez('CS-4.28.npz', cs=x)


def csReshape():
    data = np.load('CS-4.28.npz')['cs']
    sort = SortCS(Pre_Parameterization.StokesCoefficients.SortMethod,
                  Pre_Parameterization.StokesCoefficients.MaxDegree,
                  Pre_Parameterization.StokesCoefficients.MinDegree)
    c, s = sort.invert(data)

    static_c,static_s = LoadGif48().load(PathOfFiles.EIGEN6_C4).getCS(90)

    # C, S = c + static_c, s + static_s

    C, S = GeoMathKit.CS_1dTo2d(c), GeoMathKit.CS_1dTo2d(s)

    np.savez('CS2d-4.28.npz', c=C, s=S)
    pass


if __name__ == '__main__':
    # adjust()

    #
    # demo()
    # draw()
    # drawMatrix()
    # getRMS()
    # getKBRRRMS()
    # drawRMS()
    cacluateNL()
    csReshape()