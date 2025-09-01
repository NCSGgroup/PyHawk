"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/3/24 21:04
@Description:
"""

import numpy as np
import time
import functools
import numba as nb
import datetime
from scipy import linalg
import gzip, tarfile, os


class GeoMathKit:
    """
    This is a collection of math tools that will be frequently used in Gravity Solver.
    All is coded as static method for easy use
    """

    def __init__(self):
        pass

    @staticmethod
    def doodsonArguments(mjd):
        """
        This is provided by Matlab code from EOT11a
        :param mjd: TDB time in MJD convention
        :return:
        """
        dood = np.zeros((6, 1))

        # compute GMST
        Tu0 = (np.floor(mjd) - 51544.5) / 36525.0
        gmst0 = (6.0 / 24 + 41.0 / (24 * 60) + 50.54841 / (24 * 60 * 60))
        gmst0 = gmst0 + (8640184.812866 / (24 * 60 * 60)) * Tu0
        gmst0 = gmst0 + (0.093104 / (24 * 60 * 60)) * Tu0 * Tu0
        gmst0 = gmst0 + (-6.2e-6 / (24 * 60 * 60)) * Tu0 * Tu0 * Tu0
        r = 1.002737909350795 + 5.9006e-11 * Tu0 - 5.9e-15 * Tu0 * Tu0
        gmst = np.mod(2 * np.pi * (gmst0 + r * np.mod(mjd, 1)), 2 * np.pi)

        t = (mjd - 51544.5) / 365250.0

        dood[1, 0] = (218.31664562999 + (
                4812678.81195750 + (-0.14663889 + (0.00185140 + -0.00015355 * t) * t) * t) * t) * np.pi / 180
        dood[2, 0] = (280.46645016002 + (
                360007.69748806 + (0.03032222 + (0.00002000 + -0.00006532 * t) * t) * t) * t) * np.pi / 180
        dood[3, 0] = (83.35324311998 + (
                40690.13635250 + (-1.03217222 + (-0.01249168 + 0.00052655 * t) * t) * t) * t) * np.pi / 180
        dood[4, 0] = (234.95544499000 + (
                19341.36261972 + (-0.20756111 + (-0.00213942 + 0.00016501 * t) * t) * t) * t) * np.pi / 180
        dood[5, 0] = (282.93734098001 + (
                17.19457667 + (0.04568889 + (-0.00001776 + -0.00003323 * t) * t) * t) * t) * np.pi / 180
        dood[0, 0] = gmst + np.pi - dood[1, 0]

        return dood

    @staticmethod
    def nutarg(tdb_mjd, THETA):
        """
        As TT -TDB is quite small, Consequently, TT can be used in practice in place of TDB in the expressions for the
        fundamental nutation arguments. See IERS2010 page 66

        Doodson's argument calculation
        ! PURPOSE    :  COMPUTE DOODSON'S FUNDAMENTAL ARGUMENTS (BETA)
        !               AND FUNDAMENTAL ARGUMENTS FOR NUTATION (FNUT)
        !               BETA=(B1,B2,B3,B4,B5,B6)
        !               FNUT=(F1,F2,F3,F4,F5)
        !               F1=MEAN ANOMALY (MOON)
        !               F2=MEAN ANOMALY (SUN)
        !               F3=F=MOON'S MEAN LONGITUDE-LONGITUDE OF LUNAR ASC. NODE
        !               F4=D=MEAN ELONGATION OF MOON FROM SUN
        !               F5=MEAN LONGITUDE OF LUNAR ASC. NODE
        !               B2=S=F3+F5
        !               B3=H=S-F4=S-D
        !               B4=P=S-F1
        !               B5=NP=-F5
        !               B6=PS=S-F4-F2
        !               B1=THETA+PI-S
        ! PARAMETERS :.
        !        IN  :  TMJD   : TIME IN MJD                               R*8
        !               THETA  : CORRESPONDING MEAN SID.TIME GREENWICH     R*8
        !        OUT :  BETA   : DOODSON ARGUMENTS                         R*8
        !               FNUT   : FUNDAMENTAL ARGUMENTS FOR NUTATION        R*8
        ! SR CALLED  :
        ! REMARKS    :
        !
        ! AUTHOR     : FROM BERNESE5.0
        :param tdb_mjd: tdb time in MJD convention
        :param theta : CORRESPONDING MEAN SID.TIME GREENWICH
        :return:
        """
        # TIME INTERVAL (IN JUL. CENTURIES) BETWEEN TMJD AND J2000.
        TU = (tdb_mjd - 51544.5) / 36525.
        # !  FUNDAMENTAL ARGUMENTS (IN RAD)
        ROH = np.pi / 648000.
        R = 1296000.

        F1 = (485868.249036 + (1325. * R + 715923.217799902) * TU + 31.8792 * TU * TU +
              .051635 * TU * TU * TU - .0002447 * TU * TU * TU * TU) * ROH

        F2 = (1287104.79305 + (99. * R + 1292581.0480999947) * TU - .5532 * TU * TU +
              .000136 * TU * TU * TU - .00001149 * TU * TU * TU * TU) * ROH

        F3 = (335779.526232 + (1342. * R + 295262.8478000164) * TU - 12.7512 * TU * TU -
              .001037 * TU * TU * TU + .00000417 * TU * TU * TU * TU) * ROH

        F4 = (1072260.7036900001 + (1236. * R + 1105601.2090001106) * TU - 6.3706 * TU * TU +
              .006593 * TU * TU * TU - .00003169 * TU * TU * TU * TU) * ROH

        F5 = (450160.398036 - (5. * R + 482890.5431000004) * TU + 7.4722 * TU * TU +
              .007702 * TU * TU * TU - .00005939 * TU * TU * TU * TU) * ROH

        FNUT = np.array([F1, F2, F3, F4, F5])

        BETA = np.zeros((6, 1))

        BETA[1, 0] = F3 + F5
        S = BETA[1, 0]
        BETA[2, 0] = S - F4
        BETA[3, 0] = S - F1
        BETA[4, 0] = -F5
        BETA[5, 0] = S - F4 - F2
        BETA[0, 0] = THETA + np.pi - S

        return BETA

    @staticmethod
    def LegendreAndDerivatives(Nmax: int, fi):
        """
        Notice: the fi is LATITUDE rather than CO-LATITUDE
        :param Nmax: maximal degree/order
        :param fi: angle [rad]  latitude
        :return:
        pnm       normalized Legendre polynomial values
        dpnm      normalized Legendre polynomial first derivative values
        """

        pnm = np.zeros((Nmax + 1, Nmax + 1))
        dpnm = np.zeros((Nmax + 1, Nmax + 1))

        pnm[0, 0] = 1
        dpnm[0, 0] = 0
        pnm[1, 1] = np.sqrt(3) * np.cos(fi)
        dpnm[1, 1] = -np.sqrt(3) * np.sin(fi)

        # diagonal coefficients
        for i in range(2, Nmax + 1):
            pnm[i, i] = np.sqrt((2 * i + 1) / (2 * i)) * np.cos(fi) * pnm[i - 1, i - 1]
            dpnm[i, i] = np.sqrt((2 * i + 1) / (2 * i)) * ((np.cos(fi) * dpnm[i - 1, i - 1]) -
                                                           (np.sin(fi) * pnm[i - 1, i - 1]))
        # horizontal first step coefficients
        for i in range(1, Nmax + 1):
            pnm[i, i - 1] = np.sqrt(2 * i + 1) * np.sin(fi) * pnm[i - 1, i - 1]
            dpnm[i, i - 1] = np.sqrt(2 * i + 1) * ((np.cos(fi) * pnm[i - 1, i - 1]) + (np.sin(fi) * dpnm[i - 1, i - 1]))

        # horizontal second step coefficients
        for j in range(0, Nmax + 1):
            for i in range(j + 2, Nmax + 1):
                pnm[i, j] = np.sqrt((2 * i + 1) / ((i - j) * (i + j))) * \
                            ((np.sqrt(2 * i - 1) * np.sin(fi) * pnm[i - 1, j]) -
                             (np.sqrt(((i + j - 1) * (i - j - 1)) / (2 * i - 3)) * pnm[i - 2, j]))

                dpnm[i, j] = np.sqrt((2 * i + 1) / ((i - j) * (i + j))) * (
                        (np.sqrt(2 * i - 1) * np.sin(fi) * dpnm[i - 1, j]) +
                        (np.sqrt(2 * i - 1) * np.cos(fi) * pnm[i - 1, j]) -
                        (np.sqrt(((i + j - 1) * (i - j - 1)) / (2 * i - 3)) * dpnm[i - 2, j]))

        return pnm, dpnm

    @staticmethod
    def Legendre(Nmax: int, fi):
        """
        Notice: the fi is LATITUDE rather than CO-LATITUDE
        :param Nmax: maximal degree/order
        :param fi: angle [rad]  latitude
        :return:
        pnm       normalized Legendre polynomial values
        """

        pnm = np.zeros((Nmax + 1, Nmax + 1))

        pnm[0, 0] = 1
        pnm[1, 1] = np.sqrt(3) * np.cos(fi)

        # diagonal coefficients
        for i in range(2, Nmax + 1):
            pnm[i, i] = np.sqrt((2 * i + 1) / (2 * i)) * np.cos(fi) * pnm[i - 1, i - 1]

        # horizontal first step coefficients
        for i in range(1, Nmax + 1):
            pnm[i, i - 1] = np.sqrt(2 * i + 1) * np.sin(fi) * pnm[i - 1, i - 1]

        # horizontal second step coefficients
        for j in range(0, Nmax + 1):
            for i in range(j + 2, Nmax + 1):
                pnm[i, j] = np.sqrt((2 * i + 1) / ((i - j) * (i + j))) * \
                            ((np.sqrt(2 * i - 1) * np.sin(fi) * pnm[i - 1, j]) -
                             (np.sqrt(((i + j - 1) * (i - j - 1)) / (2 * i - 3)) * pnm[i - 2, j]))

        return pnm

    @staticmethod
    def CalcPolarAngles(xyz: np.ndarray):
        """
        see IERS2010 the annotation of Eq. 6.6 the detailed definition of each parameter.
        Calculate polar components from XYZ frame
        :param xyz: pos in itrf frame
        :return: geodetic coordinate: lon , lat and R
        """

        # Length of projection in x-y-plane:
        rhoSqr = np.linalg.norm(xyz[0:2])
        # Norm of vector
        r = np.linalg.norm(xyz[0:])
        # longitude
        if (xyz[0] == 0) and (xyz[1] == 0):
            phi = 0
        else:
            phi = np.arctan2(xyz[1], xyz[0])

        if phi < 0:
            phi += 2 * np.pi

        # latitude
        if (xyz[2] == 0) and (rhoSqr == 0):
            theta = 0
        else:
            theta = np.arctan2(xyz[2], rhoSqr)

        return phi, theta, r

    @staticmethod
    def Lagint(X, Y, xint, index):
        '''Normal method'''
        xnew = X[index]
        ynew = Y[index]
        yout = 0.

        for m in range(len(index)):
            term = ynew[m]
            for j in range(len(index)):
                if m != j:
                    term = term * (xint - xnew[j]) / (xnew[m] - xnew[j])
            yout = yout + term

        return yout

    @staticmethod
    def Lagint4(X, Y, xint, index):
        m = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        n = [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]

        xnew = X[index]
        ynew = Y[index]

        res = np.double((xint - xnew[n])) / (xnew[m] - xnew[n])
        a0 = res[0] * res[1] * res[2] * ynew[0]
        a1 = res[3] * res[4] * res[5] * ynew[1]
        a2 = res[6] * res[7] * res[8] * ynew[2]
        a3 = res[9] * res[10] * res[11] * ynew[3]

        return a0 + a1 + a2 + a3

    @staticmethod
    def lagint4(X, Y, xint):
        """
        This subroutine performs lagrangian interpolation
        within a set of (X,Y) pairs to give the y
        value corresponding to xint. This program uses a
        window of 4 data points to perform the interpolation.
        if the window size needs to be changed, this can be
        done by changing the indices in the do loops for
        variables m and j
        :param X:array of values of the independent variable
        :param Y:array of function values corresponding to x
        :param xint:the x-value for which estimate of y is desired

        :return: value returned to caller
        """

        assert len(X) >= 4
        assert len(X) == len(Y)

        xnew = np.append(X, xint)
        xnew2 = xnew.argsort()
        index = np.argmax(xnew2)

        if index <= 2:
            xnew3 = X[0:4]
            ynew = Y[0:4]
        elif index >= len(X) - 3:
            xnew3 = X[(len(X) - 4):]
            ynew = Y[(len(X) - 4):]
        else:
            xnew3 = X[(index - 2):(index + 2)]
            ynew = Y[(index - 2):(index + 2)]

        # pp = lagrange(xnew3, ynew)
        # the lagrange algorithm above is inaccurate

        yout = 0
        for m in range(4):
            term = ynew[m]
            for j in range(4):
                if m != j:
                    term = term * (xint - xnew3[j]) / (xnew3[m] - xnew3[j])
            yout = yout + term

        return yout

    @staticmethod
    def as2rad():
        # [seconds of arc] to [radian]
        # 1 degree = 1 h = np.pi/180 = 3600 sec
        return 2 * np.pi / 360 / 3600

    @staticmethod
    def CS_2dTo1d(CS: np.ndarray):
        """
        Transform the CS in 2-dimensional matrix to 1-dimension array
        00
        10 11
        20 21 22
        30 31 32 33          ===>         00 10 11 20 21 22 30 31 32 33 ....
        ....
        :return:
        """
        shape = np.shape(CS)
        assert len(shape) == 2

        index = np.nonzero(np.tril(np.ones(shape)))

        return CS[index]

    @staticmethod
    def CS_1dTo2d(CS: np.ndarray):
        """
        Transform the CS in 2-dimensional matrix to 1-dimension array
        00
        10 11
        20 21 22
        30 31 32 33          ===>         00 10 11 20 21 22 30 31 32 33 ....
        ....
        :return:
        """
        def index(N):
            n = (np.round(np.sqrt(2 * N))).astype(np.int64) - 1
            m = N - (n * (n + 1) / 2).astype(np.int64) - 1
            return n, m

        CS_index = np.arange(len(CS)) + 1
        n,m = index(CS_index)

        dim = index(len(CS))[0] + 1
        CS2d = np.zeros((dim, dim))
        CS2d[n, m] = CS

        return CS2d

    @staticmethod
    def arrayAdd(array1: np.ndarray, array2: np.ndarray):
        """
        This function is used for adding two arrays of different lengths.
        :param array1: one-dimension
        :param array2: one-dimension
        :return: one-dimension array
        """
        length = (len(array1), len(array2))
        res = None
        if length[0] > length[1]:
            res = array1
            res[0:length[1]] += array2
        else:
            res = array2
            res[0:length[0]] += array1

        return res

    @staticmethod
    def Sx0(S: np.ndarray):
        """
        This function is used to check the order 0 of stokes coefficient S, as it should be equal to 0
        :param S:
        :return:
        """

        '''
        00 10 11 20 21 22 30 31 32 33 => 0, 1, 3, 6, 10, 15 ... => n(n+1)/2
        '''
        L = len(S)
        N = int(np.round(np.sqrt(2 * L))) - 1

        ff = [int(n * (n + 1) / 2) for n in range(N + 1)]
        S[ff] = 0
        return S

    @staticmethod
    def getEveryDay(begin_date, end_date):
        date_list = []
        begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        while begin_date <= end_date:
            date_str = begin_date.strftime("%Y-%m-%d")
            date_list.append(date_str)
            begin_date += datetime.timedelta(days=1)
        return date_list

    @staticmethod
    def rms(x: np.ndarray):
        """

        :param x: 1-dim
        :return:
        """

        return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])

    @staticmethod
    def keepGlobal(dm_global: np.ndarray, dm_local: np.ndarray, obs: np.ndarray):
        """
        Ax + By = z
        x--global parameter
        y--local parameter
        z--obs
        Our aim: remove the local parameters and keep only the global parameters to be solved.
        :param dm_global: design matrix for the global parameters
        :param dm_local: design matrix for the local parameters
        :param obs: observations
        :return: Normal equations related only to the global parameters （Nx = l）
        """
        # time
        p = np.eye(np.shape(dm_global.T)[-1])
        I = np.eye(np.shape(dm_global.T)[-1])
        w = linalg.cholesky(p)
        B = w @ dm_local

        # n = np.shape(B)[1]
        # Q, R = np.linalg.qr(B, mode='complete')
        # Q2 = Q[:, n:]

        Q1, R = np.linalg.qr(B)
        Q2 = I - Q1 @ Q1.T
        Q2_T = Q2.T

        A = Q2_T @ w @ dm_global
        A_T = A.T

        l = Q2_T @ w @ obs

        N = A_T @ A
        L = A_T @ l

        return N, L

    @staticmethod
    def keepGlobal1(dm_global: np.ndarray, dm_local: np.ndarray, obs: np.ndarray):
        """
        Ax + By = z
        x--global parameter
        y--local parameter
        z--obs
        Our aim: remove the local parameters and keep only the global parameters to be solved.
        :param dm_global: design matrix for the global parameters
        :param dm_local: design matrix for the local parameters
        :param obs: observations
        :return: Normal equations related only to the global parameters （Nx = l）
        """
        # time
        Nxx = dm_global.T @ dm_global
        Wx = dm_global.T @ obs
        if np.shape(dm_local)[-1] == 0:
            '''No local parameters'''
            return Nxx, Wx

        Nxy = dm_global.T @ dm_local
        Nyx = dm_local.T @ dm_global
        Nyy = dm_local.T @ dm_local

        Wy = dm_local.T @ obs

        Nyy_inv = np.linalg.inv(Nyy.copy())

        tm = Nxy @ Nyy_inv

        N = Nxx - tm @ Nyx
        l = Wx - tm @ Wy
        return N, l

    def keepGlobal2(dm_global: np.ndarray, dm_local: np.ndarray, obs: np.ndarray):
        """
        Ax + By = z
        x--global parameter
        y--local parameter
        z--obs
        Our aim: remove the local parameters and keep only the global parameters to be solved.
        Avoid directly inverting local parameter matrices
        :param dm_global: design matrix for the global parameters
        :param dm_local: design matrix for the local parameters
        :param obs: observations
        :return: Normal equations related only to the global parameters （Nx = l）
        """
        # time
        Nxx = dm_global.T @ dm_global
        Wx = dm_global.T @ obs
        if np.shape(dm_local)[-1] == 0:
            '''No local parameters'''
            return Nxx, Wx

        Nxy = dm_global.T @ dm_local
        Nyx = dm_local.T @ dm_global
        Nyy = dm_local.T @ dm_local

        Wy = dm_local.T @ obs

        M_n = np.linalg.lstsq(Nyy, Nyx, rcond=None)[0]
        M_l = np.linalg.lstsq(Nyy, Wy, rcond=None)[0]

        N = Nxx - Nxy @ M_n
        l = Wx - Nxy @ M_l
        return N, l

    @staticmethod
    def solveLocal(dm_global, dm_local, obs, global_p):
        """
        Ax + By = z
        x--global parameter
        y--local parameter
        z--obs
        Our aim: given the A,B,z,x, estimate y.
        :param dm_global: design matrix for the global parameters
        :param dm_local: design matrix for the local parameters
        :param obs:
        :global_p: x
        :return: local parameters y.
        """

        if np.shape(dm_local)[-1] == 0:
            '''No local parameters'''
            return None, None

        Nyx = dm_local.T @ dm_global
        Nyy = dm_local.T @ dm_local

        Wy = dm_local.T @ obs
        Nyy_inv = np.linalg.inv(Nyy.copy())

        y = Nyy_inv@Wy - Nyy_inv @ Nyx @ global_p

        return y

    @staticmethod
    def un_gz(file_name):

        # aquire the filename and remove the postfix
        f_name = file_name.replace(".gz", "")
        # start uncompress
        g_file = gzip.GzipFile(file_name)
        # read uncompressed files and write down a copy without postfix
        open(f_name, "wb+").write(g_file.read())
        g_file.close()

    @staticmethod
    def un_tar(file_name):
        """
        Also applicable for .tgz
        :param file_name:
        :return:
        """
        tar = tarfile.open(file_name)
        names = tar.getnames()
        if os.path.isdir(file_name + '_files'):
            pass
        else:
            os.mkdir(file_name + '_files')

        for name in names:
            tar.extract(name, file_name + '_files')
        tar.close()

    @staticmethod
    def un_targz(file_name):
        """
        Also applicable for .tgz
        :param file_name:
        :return:
        """
        tar = tarfile.open(file_name)
        names = tar.getnames()
        if os.path.isdir(file_name):
            pass
        else:
            os.mkdir(file_name + '_files')

        for name in names:
            tar.extract(name, file_name + '_files')
        tar.close()

    @staticmethod
    def dayListByDay(begin, end):
        """
        get the date of every day between the given 'begin' day and 'end' day

        :param begin: year, month, day. '2009-01-01'
        :param end: year,month,day. '2010-01-01'
        :return:
        """

        daylist = []
        begin_date = datetime.datetime.strptime(begin, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d")

        while begin_date <= end_date:
            date_str = begin_date
            daylist.append(date_str)
            begin_date += datetime.timedelta(days=1)

        return daylist


if __name__ == '__main__':
    # GeoMathKit.CS_2dTo1d(np.array([[2.3, 0, 0], [1.2, 2.1, 0], [3.3, 8.9, 7.5]]))
    # GeoMathKit.arrayAdd(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([1, 2, 3, 4, 5]))
    GeoMathKit.Legendre(180, np.array([0.1, 0.2]))
