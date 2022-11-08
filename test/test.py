import h5py
import numpy as np
import time
import multiprocessing

test1 = np.ones(1000000)

test2 = [2,2,2,2,2,2,2]
filename = '2008-01-01_2008-01-31_A_0.hdf5'


def fun1(index):
    time.sleep(2)
    print("This is {}".format(index))
    r = h5py.File(filename, "r")
    data = r['acc'][:]
    print(data)


def demo1():
    time1 = time.time()
    _processes = []
    for index in range(0, 10):
        _process = multiprocessing.Process(target=fun1, args=(index,))
        _process.start()
        _processes.append(_process)
    for _process in _processes:
        _process.join()
    time2 = time.time()
    print("time consume {} s".format(time2 - time1))


def readData():
    path = "20221027-BaTmSiO.dat"

    try:
        f = open(path, 'rb')
    except FileNotFoundError as e:
        print(path)
        return None

    NumOfAcc = None
    while True:
        r = f.readline()
        content = r.decode()
        if content.split(',')[2].strip() == 'Temperature (K)':
            print(content)
            break
        # if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
        #     NumOfAcc = int(content.split(':')[1])
        # if content.strip() == 'END OF HEADER':
        #     break


if __name__ == '__main__':
    readData()
    # fun1(1)
    # time1 = time.time()
    # for i in range(0, 10):
    #     fun1(i)
    # time2 = time.time()
    # print(time2 - time1)
    # demo1()