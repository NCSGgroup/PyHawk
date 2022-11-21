import h5py
import numpy as np
import time
import multiprocessing
import src.Preference.EnumType as EnumType
import src.Preference.Pre_Parameterization as Pre_Parameterization
import matplotlib.pyplot as plt


def Sheet1():
    Data = []
    with open('Sheet1.txt', 'r', encoding='utf-8') as file:
        for line in file:
            data = line.strip("\n").split()
            Data.append(data)

    MatrixData = np.zeros((len(Data[2:]), len(Data[2:][0])))
    for i in range(len(Data[2:])):
        for j in range(len(Data[2:][i])):
            MatrixData[i][j] = float(Data[2:][i][j])

    i = 0
    j = 0
    X = np.zeros((6, len(Data[2:])))
    Y1 = np.zeros((6, len(Data[2:])))
    Y2 = np.zeros((6, len(Data[2:])))
    while i < np.shape(MatrixData)[1]:
        X[j] = MatrixData[:, i]
        Y1[j] = MatrixData[:, i+3]
        Y2[j] = MatrixData[:, i+4]
        i+=5
        j+=1

    # X = X[X != 0]
    # Y1= Y1[Y1 != 0]
    # Y2= Y2[Y2 != 0]

    fig,ax = plt.subplots(3, 2)
    k = 0
    for i in range(3):
        for j in range(2):
            ax[i][j].scatter(X[k], Y1[k], color='blue')
            ax2 = ax[i][j].twinx()
            ax2.scatter(X[k], Y2[k], color='red')
            k+=1
    plt.show()


if __name__ == '__main__':
    Sheet1()
    # fun1(1)
    # time1 = time.time()
    # for i in range(0, 10):
    #     fun1(i)
    # time2 = time.time()
    # print(time2 - time1)
    # demo1()