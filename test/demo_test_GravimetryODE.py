from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import src.Preference.EnumType as EnumType
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.ODE.Instance.GravimetryODE import GravimetryODE
from src.Interface.ArcOperation import ArcData
# from src.SecDerivative.Instance.SoleOrbit import Orbit_2nd_diff
# from src.SecDerivative.Instance.SoleOrbitAndVar import Orbit_var_2nd_diff, MatrixOrbit_var_2nd_diff
# from src.SecDerivative.Instance.DualOrbit import Orbit_2nd_diff
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib


def demo1():
    fr = Frame(EOP().load())

    # read Arc GRACE Data
    rd = ArcData()
    res = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    GPS_Time = res[4, 0]
    startPos = res[4, 1:4]
    startVel = res[4, 4:7]
    startPos = np.vstack((res[4, 1:4], res[4, 1:4]))
    startVel = np.vstack((res[4, 4:7], res[4, 4:7]))

    # instance assemble model
    assem_model = Assemble2ndDerivative()
    var_2nd_diff = Orbit_var_2nd_diff(fr)

    # ODE
    time1 = time.time()
    propagator = GravimetryODE().setInitial(GPS_Time, startPos, startVel).set2ndDerivative(var_2nd_diff)
    X_GJ, R_GJ, V_GJ = propagator.propagate()
    time2 = time.time()

    print(time2 - time1)
    # np.savez('data2.npz', time=X_GJ, R_GJ=R_GJ, V_GJ=V_GJ)
    print(X_GJ)
    print(R_GJ)
    print(V_GJ)


def demo2():
    fr = Frame(EOP().load())

    # read Arc GRACE Data
    rd = ArcData()
    resA = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    resB = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.B)
    GPS_Time = resA[4, 0]
    startPosA = resA[4, 1:4]
    startVelA = resA[4, 4:7]
    startPosB = resB[4, 1:4]
    startVelB = resB[4, 4:7]
    startPos = np.vstack((startPosA, startPosB))
    startVel = np.vstack((startVelA, startVelB))

    # instance assemble model
    var_2nd_diff = Orbit_var_2nd_diff(fr)
    assem_model = Assemble2ndDerivative()

    # ODE
    time1 = time.time()
    propagator = GravimetryODE().setInitial(GPS_Time, startPos, startVel).set2ndDerivative(var_2nd_diff)
    X_GJ, R_GJ, V_GJ = propagator.propagate()
    time2 = time.time()

    print(time2 - time1)
    # np.savez('MatrixVarDiff.npz', time=X_GJ, R_GJ=R_GJ, V_GJ=V_GJ)
    print(X_GJ)
    print(R_GJ)
    print(V_GJ)


def demo3():
    oldData = np.load('data1.npz')
    newData = np.load('data2.npz')
    oldTime = oldData['time']
    newTime = newData['time']
    oldR = oldData['R_GJ']
    newR = newData['R_GJ']
    oldV = oldData['V_GJ']
    newV = newData['V_GJ']

    diff_R = newR[9:-1] - oldR
    diff_V = newV[9:-1] - oldV

    matplotlib.rcParams.update({'font.size': 24})
    fig = plt.figure()

    plt.plot(oldTime, diff_R[:, 0], label='diff_R_X', linewidth=3)
    plt.plot(oldTime, diff_R[:, 1], label='diff_R_Y', linewidth=3)
    plt.plot(oldTime, diff_R[:, 2], label='diff_R_Z', linewidth=3)
    plt.plot(oldTime, np.sqrt(np.sum(diff_R ** 2, 1)), label='diff_R', linewidth=3)

    plt.xlim(oldTime[0], oldTime[-1])
    # plt.ylim(0, )
    plt.legend(loc='upper left', fontsize=28)
    plt.title('NewGravimetryODE VS OldGravimetryODE')
    plt.xlabel('Time (min)')
    plt.ylabel('between error relative R[m]')
    plt.grid()
    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 2))
    plt.show()

    pass


def demo4():
    VectorGNV = np.load('VectorGNV.npz')
    MatrixGNV = np.load('testMatrix.npz')
    VectorTime = VectorGNV['time']
    MatrixTime = MatrixGNV['time']
    VectorR = VectorGNV['R_GJ']
    MatrixR = MatrixGNV['R_GJ'][:, 0, :]
    VectorV = VectorGNV['V_GJ']
    MatrixV = MatrixGNV['V_GJ'][:, 0, :]

    diff_R = VectorR - MatrixR
    diff_V = VectorV - MatrixV

    matplotlib.rcParams.update({'font.size': 24})
    fig = plt.figure()

    plt.plot(VectorTime, diff_V[:, 0], label='X', linewidth=3)
    plt.plot(VectorTime, diff_V[:, 1], label='Y', linewidth=3)
    plt.plot(VectorTime, diff_V[:, 2], label='Z', linewidth=3)
    # plt.plot(VectorTime, np.sqrt(np.sum(diff_V ** 2, 1)), label='rms', linewidth=3)


    plt.xlim(VectorTime[0], VectorTime[-1])
    # plt.ylim(0, )
    plt.legend(loc='upper left', fontsize=28)
    plt.title('sat A vel MartixODE VS VectorODE')
    plt.xlabel('Time (min)')
    plt.ylabel('between error relative R[m]')
    plt.grid()
    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 2))
    plt.show()

    pass


def demo5():
    VectorGNV = np.load('VectorVarDiff.npz')
    MatrixGNV = np.load('MatrixVarDiff.npz')
    VectorTime = VectorGNV['time']
    MatrixTime = MatrixGNV['time']
    VectorR = VectorGNV['R_GJ'][:, 0:3]
    print(np.shape(MatrixGNV['R_GJ']))
    MatrixR = MatrixGNV['R_GJ'][:, :, 0:2]
    VectorV = VectorGNV['V_GJ']
    MatrixV = MatrixGNV['V_GJ']

    # print(VectorR)
    print(np.shape(MatrixR))
    print(MatrixR)
    # diff_R = VectorR - MatrixR
    # diff_V = VectorV - MatrixV
    #
    # matplotlib.rcParams.update({'font.size': 24})
    # fig = plt.figure()
    #
    # plt.plot(VectorTime, diff_V[:, 0], label='X', linewidth=3)
    # plt.plot(VectorTime, diff_V[:, 1], label='Y', linewidth=3)
    # plt.plot(VectorTime, diff_V[:, 2], label='Z', linewidth=3)
    # # plt.plot(VectorTime, np.sqrt(np.sum(diff_V ** 2, 1)), label='rms', linewidth=3)
    #
    #
    # plt.xlim(VectorTime[0], VectorTime[-1])
    # # plt.ylim(0, )
    # plt.legend(loc='upper left', fontsize=28)
    # plt.title('sat A vel MartixODE VS VectorODE')
    # plt.xlabel('Time (min)')
    # plt.ylabel('between error relative R[m]')
    # plt.grid()
    # plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 2))
    # plt.show()

    pass


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo4()
    demo5()

