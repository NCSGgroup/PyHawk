from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import src.Preference.EnumType as EnumType
from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.ConfigJson.ConfigByJson import ConfigFMbyJson
from src.ODE.GravimetryODE import GravimetryODE
from src.Interface.ArcOperation import ArcData
import json
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
    # startPos = np.vstack((res[4, 1:4], res[4, 1:4]))
    # startVel = np.vstack((res[4, 4:7], res[4, 4:7]))

    # instance assemble model
    assem_model = Assemble2ndDerivative()

    # ODE
    time1 = time.time()
    propagator = GravimetryODE(fr).setInitial(GPS_Time, startPos, startVel).setForceModel(assem_model)
    X_GJ, R_GJ, V_GJ = propagator.propagate()
    pass
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
    assem_model = Assemble2ndDerivative()

    # ODE
    time1 = time.time()
    propagator = GravimetryODE(fr).setInitial(GPS_Time, startPos, startVel).setForceModel(assem_model)
    X_GJ, R_GJ, V_GJ = propagator.propagate()
    pass
    time2 = time.time()

    print(time2 - time1)
    np.savez('MatrixGNV.npz', time=X_GJ, R_GJ=R_GJ, V_GJ=V_GJ)
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


if __name__ == '__main__':
    # demo1()
    demo2()

