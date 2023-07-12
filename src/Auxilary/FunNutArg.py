"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/2/24 下午6:36
@Description:
"""
import numpy as np


class FunNutArg:
    """
    Astronomy Fundamental arguments and Doodson (nutation) arguments
    """

    def __init__(self):
        coeff3 = np.array([
            [134.96340251 * 3600, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
            [357.52910918 * 3600, 129596581.0481, -0.5532, 0.000136, -0.00001149],
            [93.27209062 * 3600, 1739527262.8478, -12.7512, -0.001037, 0.00000417],
            [297.85019547 * 3600, 1602961601.2090, -6.3706, 0.006593, -0.00003169],
            [125.04455501 * 3600, -6962890.5431, 7.4722, 0.007702, -0.00005939]
        ])

        self.__coeff3 = np.deg2rad(coeff3 / 3600, dtype=np.float64)

        self.__flag = 0

        self.__method = self.__method3

        pass

    def __method1(self, tdb_mjd):
        """
        This is provided by Matlab code from EOT11a
        :param tdb_mjd: TDB time in MJD convention
        :return:
        """
        dood = np.zeros((6, 1))

        # compute GMST
        Tu0 = (np.floor(tdb_mjd) - 51544.5) / 36525.0
        gmst0 = (6.0 / 24 + 41.0 / (24 * 60) + 50.54841 / (24 * 60 * 60))
        gmst0 = gmst0 + (8640184.812866 / (24 * 60 * 60)) * Tu0
        gmst0 = gmst0 + (0.093104 / (24 * 60 * 60)) * Tu0 * Tu0
        gmst0 = gmst0 + (-6.2e-6 / (24 * 60 * 60)) * Tu0 * Tu0 * Tu0
        r = 1.002737909350795 + 5.9006e-11 * Tu0 - 5.9e-15 * Tu0 * Tu0
        gmst = np.mod(2 * np.pi * (gmst0 + r * np.mod(tdb_mjd, 1)), 2 * np.pi)

        t = (tdb_mjd - 51544.5) / 365250.0

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

    def __method2(self, tdb_mjd, theta):
        """

        :param tdb_mjd:
        :param theta:
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
        BETA[0, 0] = theta + np.pi - S

        return BETA

    def __method3(self, tdb_mjd, theta):
        """
        1. BenchMark test
        2. IERS Eq.(5.43)

        :param tdb_mjd:
        :param theta:
        :return:
        """
        t = (tdb_mjd - 51544.5) / 36525

        ts = np.array([
            1,
            t,
            t ** 2,
            t ** 3,
            t ** 4
        ])

        funA = np.dot(self.__coeff3, ts)

        nutarg = np.zeros(6)
        nutarg[1] = funA[2] + funA[4]
        nutarg[0] = theta + np.pi - nutarg[1]
        nutarg[2] = nutarg[1] - funA[3]
        nutarg[3] = nutarg[1] - funA[0]
        nutarg[4] = -funA[4]
        nutarg[5] = nutarg[2] - funA[1]

        return nutarg.reshape((6, 1))

    def getNutArg(self, tdb_mjd, theta):
        """
        get nutation (Doodson) argument
        :param tdb_mjd:
        :param theta: GMST, Greenwich mean sidereal time
        :return:
        """
        return self.__method(tdb_mjd, theta)


def demo1():
    fa = FunNutArg()
    res1 = fa.method3(tdb_mjd=54650.00059240741, theta=4.9101027041591285)
    res2 = fa.method2(tdb_mjd=54650.00059240741, theta=4.910135205894207)
    res3 = fa.method1(tdb_mjd=54649.99983796296)
    pass


if __name__ == '__main__':
    demo1()
