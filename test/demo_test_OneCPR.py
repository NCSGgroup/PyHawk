from src.Interface.ArcOperation import ArcSelect, ArcData
import numpy as np
import matplotlib.pyplot as plt
import src.Preference.EnumType as EnumType


def oneCPR():
    rd = ArcData()
    KBR = rd.getData(arc=0, kind=EnumType.Payload.KBR, sat=EnumType.SatID.A)

    R = np.load("adjust.npz")
    Time = R['time']
    AR = R['R2'][:, 0:3, 0]
    BR = R['R2'][:, 3:6, 0]
    baKBR = BR - AR



    plt.plot(np.array(KBR[:, 0]), KBR[:, 1] - baKBR[:, 0], label="X")
    plt.plot(np.array(KBR[:, 0]), KBR[:, 2] - baKBR[:, 1], label="Y")
    plt.plot(np.array(KBR[:, 0]), KBR[:, 3] - baKBR[:, 2], label="Z")
    plt.xlabel('GPS time [sec]', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    plt.show()


def fun(t: np.ndarray, n: np.ndarray):
    '''a+bt + (E+Ft)cosnt+(G+Ht)sinnt'''
    t = t - t[0]
    nt = n * t
    leng = np.shape(t)[0]

    term1 = np.ones(leng, dtype=np.float64)
    term2 = t
    term3 = np.cos(nt)
    term4 = t * np.cos(nt)
    term5 = np.sin(nt)
    term6 = t * np.sin(nt)

    DesignMatrix = np.array([term1, term2, term3, term4, term5, term6]).transpose()

    return DesignMatrix


if __name__ == '__main__':
    oneCPR()