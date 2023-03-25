import os
import sys
sys.path.append('../')
import multiprocessing
from src.Solver.DesignMatrix import GravityDesignMat


def demo1(start, end):

    for i in range(start, end):
        print(i)
        dm = GravityDesignMat(i).configure()
        dm.orbit_integration()
        dm.get_orbit_design_matrix()
        dm.get_sst_design_matrix()


if __name__ == '__main__':
    # demo1(15, 18)
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