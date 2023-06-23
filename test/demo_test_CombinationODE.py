from src.ODE.Common.SingleStepODE import RKN
from src.ODE.Common.MultiStepODE import GaussJackson
from src.ODE.Common.ComplexODE import ComplexODE, ConfigComplexODE
from src.Preference.EnumType import SingleStepType, MultiStepType
from src.ODE.Common.KeplerOrbit import KeplerOrbit
import matplotlib.pyplot as plt
import numpy as np


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

def demo1():
    mu = 398600.4415
    r0 = np.array([7000, 0, 0])
    v0 = np.array([0, np.sqrt(mu / np.linalg.norm(r0)), 0])
    # the unit is km
    x0 = 0

    rk = RKN()
    GJ = GaussJackson()
    propagator = ComplexODE(rk, GJ).setSecondDerivative(force_for_demo_use).setInitial(x0, r0, v0)
    t, r, v = propagator.propagate()
    print(len(t))
    print(len(r))
    print(len(v))


def demo2():
    order = 8
    mu = 398600.4415
    r0 = np.array([7000, 0, 0])
    v0 = np.array([0, np.sqrt(mu / np.linalg.norm(r0)), 0])
    # the unit is km
    x0 = 0
    h = 5
    Nstep = 4316
    config = ConfigComplexODE(SingleStepType.RKN, MultiStepType.GaussJackson).setOdePar()
    propagator = ComplexODE(config).setSecondDerivative(force_for_demo_use).setInitial(x0, r0, v0)
    X_GJ, R_GJ, v = propagator.propagate()

    kepler = KeplerOrbit().setInitial(x0, r0, v0).setArg(GM=mu, rltot=1e-9)
    X_ke, R_ke, V_ke = kepler.propagate(Nstep + int(order / 2), h)

    print(X_GJ[4:])
    print(X_ke[0:-5])
    plt.plot(np.array(X_GJ[4:]) / 60,
             np.sqrt(np.sum((np.array(R_GJ[4:])
                             - R_ke[0:-5]) ** 2, 1)) * 1e6,
             label='Gauss-Jackson 8th', linewidth=3)
    plt.show()
    np.save('data.npy',np.sqrt(np.sum((np.array(R_GJ[4:-1])
                             - R_ke[0:-6]) ** 2, 1)) * 1e6)


if __name__ == '__main__':
    demo2()