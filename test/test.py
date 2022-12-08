import h5py
import numpy as np
import time
import multiprocessing
import src.Preference.EnumType as EnumType
import src.Preference.Pre_Parameterization as Pre_Parameterization


def Sheet1():
    Data = []
    with open('Sheet1.txt', 'r', encoding='utf-8') as file:
        for line in file:
            data = line.strip("\n").split()
            Data.append(data)
    # print(Data[0])
    # print(Data[1])
    # print(Data[2])

    MatrixData = np.zeros((len(Data[2:]), len(Data[2:][0])))
    for i in range(len(Data[2:])):
        for j in range(len(Data[2:][i])):
            MatrixData[i][j] = float(Data[2:][i][j])

    print(np.shape(MatrixData))


def fitting():
    pass

if __name__ == '__main__':
    # print(Pre_Parameterization.TransitionMatrix.__name__)
    # Sheet1()
    fitting()
    # fun1(1)
    # time1 = time.time()
    # for i in range(0, 10):
    #     fun1(i)
    # time2 = time.time()
    # print(time2 - time1)
    # demo1()