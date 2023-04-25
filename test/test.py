import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import src.Preference.EnumType as EnumType
import src.Preference.Pre_Parameterization as Pre_Parameterization
import pandas as pd
from scipy.optimize import curve_fit
import scipy.optimize as optimize
from scipy.fftpack import fft, ifft
import statsmodels.api as sm
from scipy.signal import savgol_filter


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


def fun_sin(x, a0, a1, a2, a3):
    return a0 * np.sin(a1 * x + a2) + a3


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


def dealK2():
    xy = []
    with open('2k.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip("\n").split()
            xy.append([float(i) for i in data])

    x = np.array(xy)[:6000, 0] / 1000
    y = np.array(xy)[:6000, 1]
    x = x.tolist()
    y = y.tolist()

    # fs = np.fft.fftfreq(len(x), x[1] - x[0])
    # Y = abs(np.fft.fft(y))
    # freq = abs(fs[np.argmax(Y[1:]) + 1])
    # a0 = max(y) - min(y)
    # a1 = 2 * np.pi * freq
    # a2 = 0
    # a3 = np.mean(y)
    # p0 = [a0, a1, a2, a3]
    for i in range(1, 6, 1):
        p0 = [1, 0, 0, 0, 2.58931854, i]
        para, _ = optimize.curve_fit(fun_2k, x, y, p0=p0)
        print(para)
        y_fit = [fun_2k(a, *para) for a in x]
        plt.plot(x, y)
        plt.plot(x, y_fit, 'g')
        plt.show()
    # Fs = len(x)
    # k = np.arange(len(x))
    # frq = k * Fs / len(x)
    # frq1 = frq[range(int(len(x) / 2))]
    #
    # data_f = abs(np.fft.fft(y))/len(x)
    # data_f1 = data_f[range(int(len(x) / 2))]
    # plt.plot(frq1[0:10], data_f1[0:10], 'red')
    # plt.xlabel('pinlv(hz)')
    # plt.ylabel('amplitude')
    # plt.title("pinputu")
    # plt.show()


def fun_2k(H, a, b,c, d, e, f):
    return d+ b*H**3 +c*(-2*(H-e)*(a*H-e*a+f)/
                ((H-e)**2+f**2)**2-2*(H+e)*(a*H+e*a+f)/
                ((H+e)**2+f**2)**2+a/((H-e)**2+f**2)+a/((H+e)**2+f**2))


def fun(t, n):
    t = t - t[0]
    nt = n * t
    leng = np.shape(t)[0]

    term1 = np.ones(leng, dtype=np.float64)
    term2 = t
    term3 = t ** 2

    term4 = np.cos(nt)
    term5 = t * np.cos(nt)
    term6 = np.sin(nt)
    term7 = t * np.sin(nt)

    DesignMatrix = np.array([term1, term2, term3, term4, term5, term6, term7]).transpose()

    return DesignMatrix


def test2k_2():
    # 读取文件数据
    xy = []
    with open('2k.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip("\n").split()
            xy.append([float(i) for i in data])

    x = np.array(xy)[:6000, 0] / 1000
    y = np.array(xy)[:6000, 1]

    # statsmodels.api
    lowess = sm.nonparametric.lowess
    z = lowess(y, x, frac=0.02)
    plt.plot(x, y)
    plt.title('lowess')
    plt.plot(z[:, 0], z[:, 1], color='r', lw=1)

    # Savitzky-Golay filter 平滑
    # zs = savgol_filter(y, 9, 3)  # window size 51, polynomial order 3
    # plt.figure()
    # plt.plot(x, y)
    # plt.title('Savitzky-Golay filter')
    # plt.plot(x, zs, color='r', lw=1)
    plt.show()


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
    # curve_fitting()
    dealK2()
    # test2k_2()
