import os.path
import sys
sys.path.append("../")
from src.ODE.Common.MultiStepODE import MatrixGaussJackson
from src.ODE.Common.SingleStepODE import RKN
from src.ODE.Common.KeplerOrbit import KeplerOrbit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import ode
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.SecDerivative.Instance.DualOrbit import Orbit_2nd_diff as Dual_Orbit
from src.SecDerivative.Instance.SoleOrbit import Orbit_2nd_diff as Sole_Orbit
from src.Preference.Pre_ODE import ODEConfig
from src.Preference.Pre_Interface import InterfaceConfig
from src.Frame.Frame import Frame, EOP
from src.Interface.ArcOperation import ArcData
from src.Preference.EnumType import TimeFormat, Payload, SatID
from src.ODE.Common.ComplexODE import ComplexODE, ConfigComplexODE
import json
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_Frame import FrameConfig
from datetime import datetime


def force_for_demo_use(t, r, v):
    """
    Central force of the Earth
    :param t: time epoch
    :param y(r,v): state vector [pos, vel]
    :return: force (second order derivative of the pos) at given epoch [m/s^2]
    """
    mu = 398600.4415
    # State vector derivative
    a = -r / (np.linalg.norm(r,axis=1,keepdims=True) ** 3)
    # print(t,r,v)
    return a * mu


def force_for_demo_use3(t, y):
    """
    Central force of the Earth
    :param t: time epoch
    :param y: state vector [pos, vel]
    :return: velocity and force (first order derivative of the pos and velocity) at given epoch [m/s^2]
    """
    r = y[0:3]
    v = y[3:]
    mu = 398600.4415
    # State vector derivative
    a = np.divide(-r, (np.linalg.norm(r) ** 3)) * mu

    return np.append(v, a)


def demo_test_GJ():
    order = 8
    mu = 398600.4415
    r0 = np.array([7000, 0, 0])
    v0 = np.array([0, np.sqrt(mu / np.linalg.norm(r0)), 0])
    r1 = np.array([7000, 0, 0])
    v1 = np.array([0, np.sqrt(mu / np.linalg.norm(r0)), 0])
    # the unit is km
    x0 = 0
    x1 = 0
    r = np.vstack((r0, r1))
    v = np.vstack((v0, v1))
    x = np.vstack((x0, x1))
    h = 30
    Nstep = 720 * 2
    # --------------------- DOP853---------------------
    dop = ode(force_for_demo_use3).set_integrator("dop853")
    dop.set_initial_value(np.append(r0, v0), x0)
    X_dop, R_dop, V_dop = [], [], []
    for i in range(Nstep + int(order / 2)):
        dop.integrate(dop.t + h)
        X_dop.append(dop.t)
        R_dop.append(dop.y[0:3])
        V_dop.append(dop.y[4:])
    # ------------------Runge-Kutta-------------------
    rk = RKN().setInitial(r, v, x).setForce(force_for_demo_use).setRecord()
    X_RK, R_RK, V_RK = rk.propagate(Nstep + int(order / 2), h)
    X_RK = np.array(X_RK)
    X_RK = np.swapaxes(X_RK,1,0)
    R_RK = np.array(R_RK)
    R_RK = np.swapaxes(R_RK,1,0)
    V_RK = np.array(V_RK)
    # -----------------Gauss-Jackson----------------
    # get initial points using Runge-Kutta
    x1, r1, v1 = rk.propagate(int(order / 2), -h)
    x2, r2, v2 = rk.propagate(int(order / 2), h)
    x1.pop(0)
    r1.pop(0)
    v1.pop(0)
    x1.reverse()
    r1.reverse()
    v1.reverse()
    iniX = x1 + x2
    iniR = r1 + r2
    iniV = v1 + v2
    '''Gauss-Jackson integration'''
    GJ = MatrixGaussJackson().setOrder(order=8).setInitial(iniX, iniR, iniV).setForce(force_for_demo_use).setIterArgs().setRecord()
    X_GJ, R_GJ, V_GJ = GJ.propagate(Nstep, h)
    X_GJ = np.array(X_GJ)
    X_GJ = np.swapaxes(X_GJ,1,0)
    R_GJ = np.array(R_GJ)
    R_GJ = np.swapaxes(R_GJ,1,0)
    V_GJ = np.array(V_GJ)
    # compute the TRUE using kepler orbiting
    kepler = KeplerOrbit().setInitial(x0, r0, v0).setArg(GM=mu)
    X_ke, R_ke, V_ke = kepler.propagate(Nstep + int(order / 2), h)

    matplotlib.rcParams.update({'font.size': 24})
    fig = plt.figure()

    plt.plot(X_GJ[0] / 60,
             np.sqrt(np.sum((R_RK[0][(int(order / 2) + 1):] - R_ke[(int(order / 2) + 1):]) ** 2, 1)) * 1e6,
             label='RKN', linewidth=3)
    plt.plot(X_GJ[0] / 60,
             np.sqrt(np.sum((R_GJ[0] - R_ke[(int(order / 2) + 1):]) ** 2, 1)) * 1e6,
             label='Gauss-Jackson 8th', linewidth=3)
    plt.plot(X_GJ[0] / 60,
             np.sqrt(np.sum((np.array(R_dop[int(order / 2):]) - R_ke[(int(order / 2) + 1):]) ** 2, 1)) * 1e6,
             label='dopri5',linewidth=3)

    plt.xlim(0, X_GJ[0][-1] / 60)
    plt.ylim(0, )
    plt.legend(loc='upper left', fontsize=28)
    plt.title('Two-body problem: MatrixGJ8 VS Ode-dopri5')
    plt.xlabel('Time (min)')
    plt.ylabel('error relative to the kepler orbit [mm]')
    plt.grid()
    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 2))
    plt.show()

    return True


def demo_vector_GJ():
    frame_config = FrameConfig()
    frame = json.load(open('../setting/TimeCost/FrameConfig.json', 'r'))
    frame_config.__dict__ = frame

    interface_config = InterfaceConfig()
    interfaceDict = json.load(
        open('../setting/TimeCost/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict

    ode_config = ODEConfig()
    odeDict = json.load(open('../setting/TimeCost/ODEConfig.json', 'r'))
    ode_config.__dict__ = odeDict

    parameter_config = ParameterConfig()
    Parameter = json.load(open('../setting/TimeCost/ParameterConfig.json', 'r'))
    parameter_config.__dict__ = Parameter

    force_model_config = ForceModelConfig()
    fmDict = json.load(open('../setting/TimeCost/ForceModelConfig.json', 'r'))
    force_model_config.__dict__ = fmDict

    eop = EOP().configure(frameConfig=frame_config).load()
    fr = Frame(eop).configure(frameConfig=frame_config)
    rd = ArcData(interfaceConfig=interface_config)

    GNVA = rd.getData(arc=0, kind=Payload.GNV, sat=SatID.A)
    startPosA, startVelA = GNVA[4, 1:4], GNVA[4, 4:7]
    time = GNVA[4, 0]
    secDer = Sole_Orbit(fr).configures(arc=0, FMConfig=force_model_config,
                                  ParConfig=parameter_config, FrameConfig=frame_config)

    config = ConfigComplexODE().configure(ODEConfig=ode_config).setOdePar()
    propagator = ComplexODE(config).setSecondDerivative(secDer.secDerivative, ArcLen=len(GNVA[:, 0]) - 4). \
        setInitial(time, startPosA, startVelA)
    propagator.propagate()
    pass


def demo_matrix_GJ():
    frame_config = FrameConfig()
    frame = json.load(open('../setting/TimeCost/FrameConfig.json', 'r'))
    frame_config.__dict__ = frame

    interface_config = InterfaceConfig()
    interfaceDict = json.load(
        open('../setting/TimeCost/InterfaceConfig.json', 'r'))
    interface_config.__dict__ = interfaceDict

    ode_config = ODEConfig()
    odeDict = json.load(open('../setting/TimeCost/ODEConfig.json', 'r'))
    ode_config.__dict__ = odeDict

    parameter_config = ParameterConfig()
    Parameter = json.load(open('../setting/TimeCost/ParameterConfig.json', 'r'))
    parameter_config.__dict__ = Parameter

    force_model_config = ForceModelConfig()
    fmDict = json.load(open('../setting/TimeCost/ForceModelConfig.json', 'r'))
    force_model_config.__dict__ = fmDict

    eop = EOP().configure(frameConfig=frame_config).load()
    fr = Frame(eop).configure(frameConfig=frame_config)
    rd = ArcData(interfaceConfig=interface_config)

    GNVA = rd.getData(arc=0, kind=Payload.GNV, sat=SatID.A)

    startPosA, startVelA = GNVA[4, 1:4], GNVA[4, 4:7]
    time = GNVA[4, 0]
    for i in range(19):
        startPosA, startVelA = np.vstack((GNVA[4, 1:4], startPosA)), np.vstack((GNVA[4, 4:7], startVelA))

    secDer = Dual_Orbit(fr, rd).configures(arc=0, FMConfig=force_model_config,
                                       ParConfig=parameter_config, FrameConfig=frame_config)

    config = ConfigComplexODE().configure(ODEConfig=ode_config).setOdePar()
    propagator = ComplexODE(config).setSecondDerivative(secDer.secDerivative, ArcLen=len(GNVA[:, 0]) - 4). \
        setInitial(time, startPosA, startVelA)
    propagator.propagate()
    pass


def demo_time_cost():
    config = {
        "font.family": 'serif',
        "font.size": 26,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    matplotlib.rcParams.update(config)
    # matplotlib.rcParams.update({'font.size': 30})
    # plt.style.use('science')
    plt.style.use(['grid'])
    border = plt.gca()
    border.spines['top'].set_linewidth(2)  # 设置顶部边框线宽度为2
    border.spines['right'].set_linewidth(2)  # 设置右侧边框线宽度为2
    border.spines['bottom'].set_linewidth(2)  # 设置底部边框线宽度为2
    border.spines['left'].set_linewidth(2)  # 设置左侧边框线宽度为2

    namelist = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    sole = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    dual = [1.27, 1.29, 1.44, 2.00, 2.15, 2.30, 2.45, 3.03, 3.18, 3.33, 3.48, 4.03, 4.16]
    # plt.scatter(np.arange(len(rms)), rms * 1e6, s=20, color='black', label='KBRR arc RMS')
    plt.plot(namelist, sole, marker='o', linestyle='--', markersize=6, color='r', label='Traditional ODE')
    plt.plot(namelist, dual, marker='o', linestyle='--', markersize=6, color='darkblue', label='Multi-channel ODE')
    plt.xlabel('Number of satellites')
    plt.ylabel('Time cost [min]')
    # plt.ylim(0, 14)
    # plt.xticks([1354, 2708, 4062, 5416, 6770], namelist)
    plt.legend(loc='upper left', frameon=True)
    # plt.set(xlim=, ylim=[0, 30], ylabel='X [mm]')
    plt.show()
    pass


if __name__ == '__main__':
    # demo_test_GJ()
    # demo_vector_GJ()
    # demo_matrix_GJ()
    demo_time_cost()

