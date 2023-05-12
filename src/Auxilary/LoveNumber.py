from enum import Enum
import numpy as np
from scipy import interpolate
import scipy.io as scio
from src.Preference.EnumType import LoveNumberType
from src.Preference.Pre_ForceModel import ForceModelConfig


class LoveNumber:

    def __init__(self):
        self.__path = None
        self.__Nmax = None
        self.__method = None
        self.__FMConfig = None
        self.__LoveNumberConfig = None
        self.__PathOfFilesConfig = None

    def configure(self, FMConfig: ForceModelConfig):
        self.__FMConfig = FMConfig
        '''config love number'''
        self.__LoveNumberConfig = self.__FMConfig.LoveNumber()
        self.__LoveNumberConfig.__dict__.update(self.__FMConfig.LoveNumberConfig.copy())
        self.__Nmax = self.__LoveNumberConfig.Nmax
        self.__method = self.__LoveNumberConfig.method
        '''config path'''
        self.__PathOfFilesConfig = self.__FMConfig.PathOfFiles()
        self.__PathOfFilesConfig.__dict__.update(self.__FMConfig.PathOfFilesConfig.copy())
        self.__path = self.__PathOfFilesConfig.LoverNumber
        return self

    def setPath(self, fileDir:str):
        self.__path = fileDir
        return self

    def getNumber(self):
        if self.__method == LoveNumberType.Gegout97.name:
            return self.__Gegout97(Nmax=self.__Nmax)

        func = None
        if self.__method == LoveNumberType.PREM.name:
            func = self.__PREM
        elif self.__method == LoveNumberType.AOD04.name:
            func = self.__AOD04
        elif self.__method == LoveNumberType.Wang.name:
            func = self.__Wang
        elif self.__method == LoveNumberType.IERS.name:
            func = self.__IERS

        return func()

    def __PREM(self):
        """
        PREM model
        :return:
        """

        assert self.__Nmax < 200
        '''The code is unreliable once greater than 200'''

        index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          10, 12, 15, 20, 30, 40, 50,
                          70, 100, 150, 200])

        values = np.array([0.000, 0.027, -0.303, -0.194,
                           -0.132, -0.104, -0.089, -0.081,
                           -0.076, -0.072, -0.069, -0.064,
                           -0.058, -0.051, -0.040, -0.033,
                           -0.027, -0.020, -0.014, -0.010, -0.007])

        xnew = np.array(range(0, self.__Nmax + 1))

        f = interpolate.interp1d(index, values, kind="cubic")
        # ‘slinear', ‘quadratic' and ‘cubic' refer to a spline interpolation of first, second or third order)
        ynew = f(xnew)

        # print(ynew)

        return ynew

    def __AOD04(self):
        """
        Love number that was used in AOD RL04
        :return:
        """

        assert self.__Nmax <= 100

        k = np.zeros(101)

        k[0] = 0
        k[1] = 0
        k[2] = -0.308

        k[3] = -0.195
        k[4] = -0.132
        k[5] = -0.103
        k[6] = -0.089
        k[7] = -0.082
        k[8] = -0.078
        k[9] = -0.073

        for i in range(10, 18):
            k[i] = -(0.682 + 0.27 * (i - 10) / 8) / i
        for i in range(18, 32):
            k[i] = -(0.952 + 0.288 * (i - 18) / 14) / i
        for i in range(32, 56):
            k[i] = -(1.24 + 0.162 * (i - 32) / 24) / i
        for i in range(56, 101):
            k[i] = -(1.402 + 0.059 * (i - 56) / 44) / i

        # print(k[0:(self.__Nmax+1)])

        return k[0:(self.__Nmax + 1)]

    def __Wang(self):
        """
        This is a more accurate way to extract Love number, which is recommended.
        Starting from degree 0
        :return:
        """

        assert self.__Nmax <= 360

        path = self.__path + '/LoveNumber.mat'

        love = scio.loadmat(path)

        # love = scio.loadmat('raw_data/test/love')

        kl = love['love'][0:self.__Nmax, 3]

        # print(kl)

        return np.append(np.zeros(1), kl)

    def __IERS(self):
        """
        obtained from IERS2010 convention.
        :return:
        """

        assert self.__Nmax <= 30
        '''
        load love number, from 1 to 30
        '''
        Kn = np.array([
            0.000000000000000, -0.30750000000000,
            -0.19500000000000, -0.13200000000000,
            -0.10320000000000, -0.89166666666670e-1,
            -0.81710392640550e-1, -0.75500000000000e-1,
            -0.71685683412260e-1, -0.68200000000000e-1,
            -0.65980069344540e-1, -0.63812455645590e-1,
            -0.61732085548940e-1, -0.59754188127910e-1,
            -0.57883368816860e-1, -0.56118520212550e-1,
            -0.54455544917280e-1, -0.52888888888890e-1,
            -0.51529657180340e-1, -0.50236923831480e-1,
            -0.49007643741670e-1, -0.47838465083770e-1,
            -0.46725942423010e-1, -0.45666666666670e-1,
            -0.44657342166760e-1, -0.43694830109180e-1,
            -0.42776170404080e-1, -0.41898589949110e-1,
            -0.41059502372580e-1, -0.40256502584650e-1
        ])

        return np.append(np.zeros(1), Kn[0:self.__Nmax])

    def __Gegout97(self, Nmax):

        assert Nmax <= 1024

        with open(self.__path + '/Load_Love2_CM.txt') as load_f:
            contents = load_f.readlines()

        Kn = np.zeros(Nmax+1)

        for i in range(13, 14+Nmax):
            res = np.array(contents[i].split())
            Kn[int(res[0])] = float(res[3].replace('D','e'))

        return Kn
