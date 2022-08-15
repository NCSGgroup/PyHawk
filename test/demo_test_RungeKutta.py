from src.ODE.RungeKutta import RungeKutta
from src.Preference.EnumType import RKOrder
import numpy as np


def force_for_demo_use2(t, y):
    """
    Central force of the Earth
    :param t: time epoch
    :param y: state vector [pos, vel]
    :return: velocity and force (first order derivative of the pos and velocity) at given epoch [m/s^2]
    """
    r = y[0:3]
    v = y[3:]
    # State vector derivative
    a = -r / (np.linalg.norm(r) ** 3) * 398600.4415
    return np.append(v, a)


def demo1():
    yref = np.array([-5172.89038285,
                     -4716.05821497,
                     -0,
                     5.08394666,
                     -5.57641521,
                     0])
    r0 = np.array([7000, 0, 0])
    v0 = np.array([0, np.sqrt(398600.4415 / np.linalg.norm(r0)), 0])
    x0 = 0


    rk = RungeKutta().setRecord()
    print("Test case for %s" % RKOrder.rk4.name)
    rk.setInitial(r0, v0, x0).setForce(force_for_demo_use2)
    x, r, v = rk.propagate()
    # print(r / (np.linalg.norm(r)))
    print(x, r, v)
    print("The accuracy is %e" % (np.linalg.norm(yref[0:3] - r)))


if __name__ == '__main__':
    demo1()