"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/3/24 上午11:36
@Description:
"""
import numpy as np
from src.Auxilary.KeplerOrbitElements import OrbitalElements


def demo1():
    oe = OrbitalElements(GM=3.986004418e14)
    par = np.array([-6045000, -3490000, 2500000, -3.457e3, 6.618e3, 2.533e3], dtype=np.float64)
    par = np.vstack((par, par))
    oe.setState(par[:, 0], par[:, 1], par[:, 2], par[:, 3], par[:, 4], par[:, 5])
    kep = oe.getKepler()
    n = oe.getMeanMotion()
    print(n)
    pass


def demo2():
    oe = OrbitalElements(GM=3.986004418e14)
    par = np.array([8.378e6, 0.000000023e6, 0.000000747000920e6,
                    0.000000865020084e6, 0.000005001014079e6, 0.000000389252913e6], dtype=np.float64)
    par = np.vstack((par, par))
    oe.setKepler(par[:, 0], par[:, 1], par[:, 2], par[:, 3], par[:, 4], par[:, 5])
    state = oe.getState()
    pass


if __name__ == '__main__':
    demo1()
    # demo2()
