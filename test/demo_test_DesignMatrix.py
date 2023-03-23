import os
import sys
sys.path.append('../')
import multiprocessing
from src.Solver.DesignMatrix import GravityDesignMat


def demo1(start, end):

    for i in range(start, end):
        dm = GravityDesignMat(i).configure()
        dm.orbit_integration()
        dm.get_orbit_design_matrix()
        dm.get_sst_design_matrix()


if __name__ == '__main__':
    demo1(0, 1)
    # p1 = multiprocessing.Process(target=demo1, args=(0, 18))
    # p2 = multiprocessing.Process(target=demo1, args=(18, 36))
    # p3 = multiprocessing.Process(target=demo1, args=(36, 54))
    # p4 = multiprocessing.Process(target=demo1, args=(54, 72))
    # p5 = multiprocessing.Process(target=demo1, args=(72, 90))
    # p6 = multiprocessing.Process(target=demo1, args=(90, 108))
    #
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    #
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()