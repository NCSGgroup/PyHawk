from src.ODE.SingleStepODE import RKN
from src.ODE.MultiStepODE import GaussJackson
from src.ODE.CombinationODE import Propagator, PropagatorConfig
from src.Preference.EnumType import SingleStepType, MultiStepType
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
    propagator = Propagator(rk, GJ).setForceModel(force_for_demo_use).setInitial(x0, r0, v0)
    t, r, v = propagator.propagate()
    print(len(t))
    print(len(r))
    print(len(v))

def demo2():
    mu = 398600.4415
    r0 = np.array([7000, 0, 0])
    v0 = np.array([0, np.sqrt(mu / np.linalg.norm(r0)), 0])
    # the unit is km
    x0 = 0

    config = PropagatorConfig().setODE(SingleStepType.RKN, MultiStepType.GaussJackson)
    propagator = Propagator(config).setForceModel(force_for_demo_use).setInitial(x0, r0, v0)
    t, r, v = propagator.propagate()
    print(len(t))
    print(len(r))
    print(len(v))


if __name__ == '__main__':
    demo2()