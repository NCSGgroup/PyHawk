import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import src.Preference.EnumType as EnumType
import src.Preference.Pre_Parameterization as Pre_Parameterization
import pandas as pd
from scipy.optimize import curve_fit


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
    str = EnumType.SatID.A.name
    sat = EnumType.SatID.B.name
    str = str + sat
    print(str)


def test2():
    arr1 = np.array([[1,3,4],
                    [2,3,4],
                    [2,3,4]])
    print(arr1[:,0])
    print(arr1[:,1])
    print(arr1[:,2])
    print(np.sqrt(arr1[:,0]**2 + arr1[:,1]**2 + arr1[:,2]**2))


def fun_poly_7(x, b1, b2, b3):
    return b1*np.power(x, 3) + b2*np.power(x, 5) + b3*np.power(x, 7)


def curve_fitting():
    xy = []
    with open('curvefitt.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip("\n").split()
            xy.append([float(i) for i in data])

    x = np.array(xy)[:, 0]
    y = np.array(xy)[:, 1]

    z = np.polyfit(x, y, 7)
    p = np.poly1d(z)
    values = p(x)
    popt, pcov = curve_fit(fun_poly_7, x, y)

    plt.scatter(x, y, label='original values')
    # plt.plot(x, values, label='curvefit values')
    plt.plot(x, [fun_poly_7(i, *popt) for i in x], label='curvefit value')
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    # print(Pre_Parameterization.TransitionMatrix.__name__)
    # Sheet1()
    # fitting()
    # fun1(1)
    # test2()
    # time1 = time.time()
    # for i in range(0, 10):
    #     fun1(i)
    # time2 = time.time()
    # print(time2 - time1)
    # demo1()
    # path = "Yb-60K拟合.txt"
    # data = pd.read_csv(path)
    # print(data)
    curve_fitting()