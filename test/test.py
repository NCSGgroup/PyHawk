import h5py
import numpy as np
import time
import multiprocessing
import src.Preference.EnumType as EnumType
import src.Preference.Pre_Parameterization as Pre_Parameterization



if __name__ == '__main__':
    for term in EnumType.ParaType:
        str = term
        print(str)
    # fun1(1)
    # time1 = time.time()
    # for i in range(0, 10):
    #     fun1(i)
    # time2 = time.time()
    # print(time2 - time1)
    # demo1()