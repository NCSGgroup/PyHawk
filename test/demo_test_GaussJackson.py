from src.ODE.MultiStepODE import GaussJackson
from src.ODE.SingleStepODE import RKN
from src.ODE.KeplerOrbit import KeplerOrbit
from src.Preference.EnumType import SingleStepType
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import ode
import time as Ti


def force_for_demo_use(t, r, v):
    """
    Central force of the Earth
    :param t: time epoch
    :param y(r,v): state vector [pos, vel]
    :return: force (second order derivative of the pos) at given epoch [m/s^2]
    """
    mu = 398600.4415
    # State vector derivative
    a = -r / (np.linalg.norm(r) ** 3)
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
    # the unit is km
    x0 = 0
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
    rk = RKN().setInitial(r0, v0, x0).setForce(force_for_demo_use).setRecord()
    X_RK, R_RK, V_RK = rk.propagate(Nstep + int(order / 2), h)

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
    # get initial points using dopri5
    # dop.set_initial_value(np.append(r0, v0), x0)
    # x1, r1, v1 = [], [], []
    # x2, r2, v2 = [], [], []
    #
    # for i in range(4):
    #     dop.integrate(dop.t + h)
    #     x1.append(dop.t)
    #     r1.append(dop.y[0:3])
    #     v1.append(dop.y[3:])
    #
    # dop.set_initial_value(np.append(r0, v0), x0)
    # x2.append(dop.t)
    # r2.append(dop.y[0:3])
    # v2.append(dop.y[3:])
    #
    # for i in range(4):
    #     dop.integrate(dop.t - h)
    #     x2.append(dop.t)
    #     r2.append(dop.y[0:3])
    #     v2.append(dop.y[3:])
    #
    # x2.reverse()
    # r2.reverse()
    # v2.reverse()
    # iniX = x2 + x1
    # iniR = r2 + r1
    # iniV = v2 + v1

    '''Gauss-Jackson integration'''
    # GJ = GaussJackson(order=order).setInit(force_for_demo_use, x0, r0, v0)
    # X_GJ, R_GJ, V_GJ = GJ.propagate(h, Nstep)
    GJ = GaussJackson().setOrder().setInitial(iniX, iniR, iniV).setForce(force_for_demo_use).setIterArgs().setRecord()
    X_GJ, R_GJ, V_GJ = GJ.propagate(Nstep, h)
    # compute the TRUE using kepler orbiting
    kepler = KeplerOrbit().setInitial(x0, r0, v0).setArg(GM=mu)
    X_ke, R_ke, V_ke = kepler.propagate(Nstep + int(order / 2), h)

    matplotlib.rcParams.update({'font.size': 24})
    fig = plt.figure()

    # print()
    plt.plot(np.array(X_GJ) / 60,
             np.sqrt(np.sum((np.array(R_RK[(int(order / 2) + 1):])
             - R_ke[(int(order / 2) + 1):]) ** 2, 1)) * 1e6,
             label='RKN', linewidth=3)
    # plt.plot(np.array(X_GJ)/60, np.max(np.abs(np.array(R_GJ) - R_ke[5:]), 1)*1e6,  label='Gauss-Jackson 8th', linewidth=3)

    plt.plot(np.array(X_GJ) / 60,
             np.sqrt(np.sum((np.array(R_GJ)
            - R_ke[(int(order / 2) + 1):]) ** 2, 1)) * 1e6,
             label='Gauss-Jackson 8th', linewidth=3)
    plt.plot(np.array(X_GJ) / 60,
             np.sqrt(np.sum((np.array(R_dop[int(order / 2):])
             -R_ke[(int(order / 2) + 1):]) ** 2, 1)) * 1e6,
             label='dopri5',linewidth=3)

    a = np.load('data.npy')

    plt.plot(np.linspace(2.5,722,1439),np.load('data.npy'),
             label='ComplexODE',linewidth=3)
    plt.xlim(0, X_GJ[-1] / 60)
    plt.ylim(0, )
    plt.legend(loc='upper left', fontsize=28)
    plt.title('ComplexODE VS GJ8')
    plt.xlabel('Time (min)')
    plt.ylabel('error relative to the kepler orbit [mm]')
    plt.grid()
    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 2))
    plt.show()

    return True

if __name__ == '__main__':
    demo_test_GJ()