from src.ForceModel.BaseGravCS import BaseGravCS
from src.Auxilary.FunNutArg import FunNutArg
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Frame.PlanetEphemerides import PlanetEphemerides
from src.Preference.Pre_ForceModel import ForceModelConfig
import numpy as np


class AbsSolidTide(BaseGravCS):

    def __init__(self):
        super(AbsSolidTide, self).__init__()
        self._ng = FunNutArg()
        self._GM_Earth = None
        self._GM_Sun = None
        self._GM_Moon = None
        self._Radius_Earth = None
        self._FMConfig = None
        self._SolidTideConfig = None

    def configure(self, FMConfig: ForceModelConfig):
        return self


class SolidTideIERS2010(AbsSolidTide):

    def __init__(self, pe: PlanetEphemerides):
        super(SolidTideIERS2010, self).__init__()
        self.__doodA = None
        self.__doodB = None
        self.__doodC = None
        self.__love = None
        self.__isZeroTide = None
        self.__prepare()
        self.__pe = pe

    def configure(self, FMConfig: ForceModelConfig):
        self._FMConfig = FMConfig
        '''config SolidTideConfig'''
        self._SolidTideConfig = self._FMConfig.SolidTide()
        self._SolidTideConfig.__dict__.update(self._FMConfig.SolidTideConfig.copy())
        self._GM_Earth = self._SolidTideConfig.GM_Earth
        self._GM_Sun = self._SolidTideConfig.GM_Sun
        self._GM_Moon = self._SolidTideConfig.GM_Moon
        self._Radius_Earth = self._SolidTideConfig.Radius_Earth
        self.__isZeroTide = self._SolidTideConfig.isZeroTide
        return self

    def getCS(self):
        """
        Note!! The position vector has to be given in ITRS frame
        :param pos_sun:POSITION VECTOR OF SUN [meters], ITRS
        :param pos_moon:POSITION VECTOR OF MOON [meters], ITRS
        :param tdb_mjd: TIME IN MJD (TDB TIME)
        :param theta: CORRESPONDING MEAN SID.TIME GREENWICH. = GMST
        :param isZeroTide: False: TIDE FREE, True: ZERO TIDE
        :return: C, S coefficients in one dimension format up to degree/order 5
        """
        pos_sun = self.__pe.getPosPlanets()['Sun']
        pos_moon = self.__pe.getPosPlanets()['Moon']
        rm = self._time.getRotationMatrix
        pos_sun = self._time.PosGCRS2ITRS(pos_sun, rm)
        pos_moon = self._time.PosGCRS2ITRS(pos_moon, rm)

        tdb_mjd = self._time.getTDB_mjd
        theta = self._time.getTheta
        index = self.__getIndex
        (K20, RK21, IK21, RK22, IK22, K30, K31, K32, K33, K20P, K21P, K22P) = self.__love

        maxDeg = 4
        Cpot = np.zeros(int((maxDeg + 1) * (maxDeg + 2) / 2))
        Spot = np.zeros(int((maxDeg + 1) * (maxDeg + 2) / 2))

        '''
        STEP 1: frequency-independent, Eq 6.6 and Eq. 6.7
        '''
        for planet in ['Moon', 'Sun']:

            if planet == 'Sun':
                GM = self._GM_Sun
                pos = pos_sun
            else:
                GM = self._GM_Moon
                pos = pos_moon
            lon, lat, r = GeoMathKit.CalcPolarAngles(pos)
            GM_ratio = GM / self._GM_Earth
            R_ratio = self._Radius_Earth / r
            # n = 2
            factor1 = GM_ratio / 5 * R_ratio * R_ratio * R_ratio
            # n = 3
            factor2 = GM_ratio / 7 * R_ratio * R_ratio * R_ratio * R_ratio

            # calculate the Pnm up to 3
            Pnm = GeoMathKit.Legendre(3, lat)

            # n=2: Anelastic Earth
            Cpot[index(2, 0)] += K20 * Pnm[2, 0] * factor1
            Cpot[index(2, 1)] += factor1 * Pnm[2, 1] * (RK21 * np.cos(lon) + IK21 * np.sin(lon))
            Spot[index(2, 1)] += factor1 * Pnm[2, 1] * (RK21 * np.sin(lon) - IK21 * np.cos(lon))
            Cpot[index(2, 2)] += factor1 * Pnm[2, 2] * (RK22 * np.cos(lon * 2) + IK22 * np.sin(lon * 2))
            Spot[index(2, 2)] += factor1 * Pnm[2, 2] * (RK22 * np.sin(lon * 2) - IK22 * np.cos(lon * 2))

            # n=3: elastic Earth
            Cpot[index(3, 0)] += K30 * Pnm[3, 0] * factor2
            Cpot[index(3, 1)] += K31 * factor2 * Pnm[3, 1] * np.cos(lon)
            Spot[index(3, 1)] += K31 * factor2 * Pnm[3, 1] * np.sin(lon)
            Cpot[index(3, 2)] += K32 * factor2 * Pnm[3, 2] * np.cos(lon * 2)
            Spot[index(3, 2)] += K32 * factor2 * Pnm[3, 2] * np.sin(lon * 2)
            Cpot[index(3, 3)] += K33 * factor2 * Pnm[3, 3] * np.cos(lon * 3)
            Spot[index(3, 3)] += K33 * factor2 * Pnm[3, 3] * np.sin(lon * 3)

            # n=4:
            Cpot[index(4, 0)] += K20P * factor1 * Pnm[2, 0]
            Cpot[index(4, 1)] += K21P * factor1 * Pnm[2, 1] * np.cos(lon)
            Spot[index(4, 1)] += K21P * factor1 * Pnm[2, 1] * np.sin(lon)
            Cpot[index(4, 2)] += K22P * factor1 * Pnm[2, 2] * np.cos(lon * 2)
            Spot[index(4, 2)] += K22P * factor1 * Pnm[2, 2] * np.sin(lon * 2)

        '''
        Step 2: frequency dependent terms for n=2
        '''
        doodsonArg = self._ng.getNutArg(tdb_mjd, theta)

        # Corrections for C20 with 21 tides recorded in Table 6.5b
        doodsonMatrix = self.__doodB[:, 0:6]
        ip = self.__doodB[:, 6]
        op = self.__doodB[:, 7]
        thetaf = np.matmul(doodsonMatrix, doodsonArg).flatten()
        dC20 = np.cos(thetaf) * ip - np.sin(thetaf) * op
        Cpot[index(2, 0)] += np.sum(dC20, 0) * 1e-12

        # Corrections for C21 with 48 tides recorded in Table 6.5a
        doodsonMatrix = self.__doodA[:, 0:6]
        ip = self.__doodA[:, 6]
        op = self.__doodA[:, 7]
        thetaf = np.matmul(doodsonMatrix, doodsonArg).flatten()
        dC21 = np.sin(thetaf) * ip + np.cos(thetaf) * op
        dS21 = np.cos(thetaf) * ip - np.sin(thetaf) * op
        Cpot[index(2, 1)] += np.sum(dC21, 0) * 1e-12
        Spot[index(2, 1)] += np.sum(dS21, 0) * 1e-12

        # Corrections for C22 with 2 tides recorded in Table 6.5c
        doodsonMatrix = self.__doodC[:, 0:6]
        amp = self.__doodC[:, 6]
        thetaf = np.matmul(doodsonMatrix, doodsonArg).flatten()
        dC22 = np.cos(thetaf) * amp
        dS22 = np.sin(thetaf) * amp * (-1)
        Cpot[index(2, 2)] += np.sum(dC22, 0) * 1e-12
        Spot[index(2, 2)] += np.sum(dS22, 0) * 1e-12

        # if self.__isZeroTide:
        #     Cpot[index(2, 0)] += 4.4228e-8 * 0.31460 * K20

        if self.__isZeroTide:
            Cpot[index(2, 0)] += 4.201e-9

        return Cpot, Spot

    def __prepare(self):
        """
        Preprocessor for once time
        :return:
        """

        # DoodsonNumber      Amp(R)       Amp(I)

        # Table 6.5a
        #
        coeff_1 = [
            [1, -3, 0, 2, 0, 0, -0.1, 0.],
            [1, -3, 2, 0, 0, 0, -0.1, 0.],
            [1, -2, 0, 1, -1, 0, -0.1, 0.],
            [1, -2, 0, 1, 0, 0, -0.7, 0.1],
            [1, -2, 2, -1, 0, 0, -0.1, 0.],
            [1, -1, 0, 0, -1, 0, -1.3, 0.1],
            [1, -1, 0, 0, 0, 0, -6.8, 0.6],
            [1, -1, 2, 0, 0, 0, 0.1, 0.],
            [1, 0, -2, 1, 0, 0, 0.1, 0.],
            [1, 0, 0, -1, -1, 0, 0.1, 0.],
            [1, 0, 0, -1, 0, 0, 0.4, 0.],
            [1, 0, 0, 1, 0, 0, 1.3, -0.1],
            [1, 0, 0, 1, 1, 0, 0.3, 0.],
            [1, 0, 2, -1, 0, 0, 0.3, 0.],
            [1, 0, 2, -1, 1, 0, 0.1, 0.],
            [1, 1, -3, 0, 0, 1, -1.9, 0.1],
            [1, 1, -2, 0, -1, 0, 0.5, 0.],
            [1, 1, -2, 0, 0, 0, -43.4, 2.9],
            [1, 1, -1, 0, 0, -1, 0.6, 0.],
            [1, 1, -1, 0, 0, 1, 1.6, -0.1],
            [1, 1, 0, -2, -1, 0, 0.1, 0.],
            [1, 1, 0, 0, -2, 0, 0.1, 0.],
            [1, 1, 0, 0, -1, 0, -8.8, 0.5],
            [1, 1, 0, 0, 0, 0, 470.9, -30.2],
            [1, 1, 0, 0, 1, 0, 68.1, -4.6],
            [1, 1, 0, 0, 2, 0, -1.6, 0.1],
            [1, 1, 1, -1, 0, 0, 0.1, 0.],
            [1, 1, 1, 0, -1, -1, -0.1, 0.],
            [1, 1, 1, 0, 0, -1, -20.6, -0.3],
            [1, 1, 1, 0, 0, 1, 0.3, 0.],
            [1, 1, 1, 0, 1, -1, -0.3, 0.],
            [1, 1, 2, -2, 0, 0, -0.2, 0.],
            [1, 1, 2, -2, 1, 0, -0.1, 0.],
            [1, 1, 2, 0, 0, 0, -5., 0.3],
            [1, 1, 2, 0, 1, 0, 0.2, 0.],
            [1, 1, 3, 0, 0, -1, -0.2, 0.],
            [1, 2, -2, 1, 0, 0, -0.5, 0.],
            [1, 2, -2, 1, 1, 0, -0.1, 0.],
            [1, 2, 0, -1, -1, 0, 0.1, 0.],
            [1, 2, 0, -1, 0, 0, -2.1, 0.1],
            [1, 2, 0, -1, 1, 0, -0.4, 0.],
            [1, 3, -2, 0, 0, 0, -0.2, 0.],
            [1, 3, 0, -2, 0, 0, -0.1, 0.],
            [1, 3, 0, 0, 0, 0, -0.6, 0.],
            [1, 3, 0, 0, 1, 0, -0.4, 0.],
            [1, 3, 0, 0, 2, 0, -0.1, 0.],
            [1, 4, 0, -1, 0, 0, -0.1, 0.],
            [1, 4, 0, -1, 1, 0, -0.1, 0.]
        ]

        # table 6.5b
        coeff_2 = [
            [0, 0, 0, 0, 1, 0, 16.6, -6.7],
            [0, 0, 0, 0, 2, 0, -0.1, 0.1],
            [0, 0, 1, 0, 0, -1, -1.2, 0.8],
            [0, 0, 2, 0, 0, 0, -5.5, 4.3],
            [0, 0, 2, 0, 1, 0, 0.1, -0.1],
            [0, 0, 3, 0, 0, -1, -0.3, 0.2],
            [0, 1, -2, 1, 0, 0, -0.3, 0.7],
            [0, 1, 0, -1, -1, 0, 0.1, -0.2],
            [0, 1, 0, -1, 0, 0, -1.2, 3.7],
            [0, 1, 0, -1, 1, 0, 0.1, -0.2],
            [0, 1, 0, 1, 0, 0, 0.1, -0.2],
            [0, 2, -2, 0, 0, 0, 0., 0.6],
            [0, 2, 0, -2, 0, 0, 0., 0.3],
            [0, 2, 0, 0, 0, 0, 0.6, 6.3],
            [0, 2, 0, 0, 1, 0, 0.2, 2.6],
            [0, 2, 0, 0, 2, 0, 0., 0.2],
            [0, 3, -2, 1, 0, 0, 0.1, 0.2],
            [0, 3, 0, -1, 0, 0, 0.4, 1.1],
            [0, 3, 0, -1, 1, 0, 0.2, 0.5],
            [0, 4, -2, 0, 0, 0, 0.1, 0.2],
            [0, 4, 0, -2, 0, 0, 0.1, 0.1]
        ]

        # table 6.5c
        coeff_3 = [[2, -1, 0, 1, 0, 0, -0.3],
                   [2, 0, 0, 0, 0, 0, -1.2]]

        # Table 6.3 Love number
        # elastic Earth
        RK20 = .30190e0
        RK21 = .29830e0
        RK22 = .30102

        IK21 = -0.00144
        IK22 = -0.0013

        K30 = .093
        K31 = .093
        K32 = .093
        K33 = .094
        # Anelastic Earth)
        K20P = -.00089
        K21P = -.00080
        K22P = -.00057

        coeff_4 = [RK20, RK21, IK21, RK22, IK22, K30, K31, K32, K33, K20P, K21P, K22P]

        self.__doodA = np.array(coeff_1)
        self.__doodB = np.array(coeff_2)
        self.__doodC = np.array(coeff_3)
        self.__love = coeff_4

    def __getIndex(self, n, m):
        assert m <= n
        return int(n * (n + 1) / 2 + m)