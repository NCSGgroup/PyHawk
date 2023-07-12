from src.ForceModel.BaseGravCS import BaseGravCS
from src.Auxilary.LoveNumber import LoveNumber
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Frame.Frame import Frame
import numpy as np


class AbsPoleTide(BaseGravCS):

    def __init__(self):
        super(AbsPoleTide, self).__init__()
        self._FMConfig = None
        self._PoleTideConfig = None
        self._PathConfig = None

    def configure(self, FMConfig: ForceModelConfig):
        self._FMConfig = FMConfig
        return self


class PoleTideIERS2010(AbsPoleTide):

    def __init__(self):
        super(PoleTideIERS2010, self).__init__()
        self.__m1 = None
        self.__m2 = None
        self.__Rn = None
        self.__maxdegree = None
        self.__poleTideModel = None
        self.__RGAMA, self.__IGAMA = None, None
        self.__simpleOcean = None
        self.__version = None
        self.__kind = None
        self.__fileLove = None
        self.__fileDir = None
        self.__loveNum = None
    
    def configure(self, FMConfig: ForceModelConfig):
        self._FMConfig = FMConfig
        '''config pole tide'''
        self._PoleTideConfig = self._FMConfig.PoleTide()
        self._PoleTideConfig.__dict__.update(self._FMConfig.PoleTideConfig.copy())

        self.__simpleOcean = self._PoleTideConfig.SimpleOcean
        self.__version = self._PoleTideConfig.Polar2wobble
        kind = self._PoleTideConfig.Kind
        self.__kind = [key for key, value in kind.items() if value]
        self.__maxdegree = self._PoleTideConfig.MaxDegreeOfOcean
        self.__loveNum = self._PoleTideConfig.LoveNumberType
        '''config path'''
        self._PathConfig = self._FMConfig.PathOfFiles()
        self._PathConfig.__dict__.update(self._FMConfig.PathOfFilesConfig.copy())

        self.__fileLove = self._PathConfig.LoverNumber
        self.__fileDir = self._PathConfig.PoleTide
        self.__poleTideModel = self._PathConfig.poleTideModel
        return self

    def load(self, **kwargs):
        """
        Load all necessary files and preprocess coefficients that will be used later on
        :param fileLove: directory of the love number file.
        :param loveNum: which love number to be extracted
        :param maxDegreeOfOcean: maximal degree of ocean pole tide expected to be output. <=360
        :param fileDir: directory of "desaiscopolecoeff.txt" provided by IERS, and the love number file.
        :return:
        """
        '''
        get load love number, from 1 to 30
        '''
        LN = LoveNumber().configure(self._FMConfig).setPath(self.__fileLove)
        Kn = LN.getNumber()
        '''start from degree 1'''
        Kn = Kn[1:]

        '''
        Load constants, see 6.23b
        '''
        OMEGA = kwargs.get('Omeag', 7.292115e-05)
        GM = kwargs.get('GM', 3.986004418e14)
        AE = kwargs.get('R', 6378136.6)
        RHOW = kwargs.get('rho', 1025)
        G = kwargs.get('G', 6.673e-11)
        GEM = kwargs.get('g', 9.7803278)

        self.__RGAMA, self.__IGAMA = 0.6870, 0.0036

        factor = 4. * np.pi * OMEGA ** 2 * AE ** 4 / GM * G * RHOW / GEM

        self.__Rn = np.zeros(int(self.__maxdegree * (self.__maxdegree + 1) / 2 + self.__maxdegree))
        for n in range(1, self.__maxdegree + 1):
            for m in range(n + 1):
                self.__Rn[int(n * (n + 1) / 2 + m - 1)] = factor * (1. + Kn[n - 1]) / (2. * n + 1.)

        '''
        Load Desais coefficients
        '''
        with open(self.__fileDir + '/' + self.__poleTideModel) as f:
            content = f.readlines()
            pass

        n, m, AnmR, BnmR, AnmI, BnmI = [], [], [], [], [], []
        for item in content[1:]:
            value = item.split()

            n.append(value[0])
            m.append(value[1])
            AnmR.append(value[2])
            BnmR.append(value[3])
            AnmI.append(value[4])
            BnmI.append(value[5])

        n = np.array(n).astype(np.int64)
        m = np.array(m).astype(np.int64)
        AnmR = np.array(AnmR).astype(np.float64)
        BnmR = np.array(BnmR).astype(np.float64)
        AnmI = np.array(AnmI).astype(np.float64)
        BnmI = np.array(BnmI).astype(np.float64)

        self.__AnmR = np.zeros(len(n))
        self.__BnmR = np.zeros(len(n))
        self.__AnmI = np.zeros(len(n))
        self.__BnmI = np.zeros(len(n))

        # start from (1,0)
        self.__AnmR[(n * (n + 1) / 2 + m - 1).astype(np.int64)] = AnmR
        self.__BnmR[(n * (n + 1) / 2 + m - 1).astype(np.int64)] = BnmR
        self.__AnmI[(n * (n + 1) / 2 + m - 1).astype(np.int64)] = AnmI
        self.__BnmI[(n * (n + 1) / 2 + m - 1).astype(np.int64)] = BnmI

        index = int(self.__maxdegree * (self.__maxdegree + 1) / 2 + self.__maxdegree)
        self.__AnmR = self.__AnmR[0: index]
        self.__AnmI = self.__AnmI[0: index]
        self.__BnmR = self.__BnmR[0: index]
        self.__BnmI = self.__BnmI[0: index]

        '''
        load m1m2 interpolation/extrapolation coefficients, see IERS2010 V1.0.0 : section 7.1.4
        '''
        Xcoeff_bf2010 = np.array([55.974, 1.8243, 0.18413, 0.007024])
        Ycoeff_bf2010 = np.array([346.346, 1.7896, -0.10729, -0.000908])
        Xcoeff_af2010 = np.array([23.513, 7.6141, 0, 0])
        Ycoeff_af2010 = np.array([358.891, -0.6287, 0, 0])
        self.__m1m2_coeff_v1 = (Xcoeff_bf2010, Ycoeff_bf2010, Xcoeff_af2010, Ycoeff_af2010)

        '''
        load m1m2 interpolation coefficients according to IERS2010 V1.2.0 : section 7.1.4
        '''
        Xcoeff_v2 = np.array([55.0, 1.677])
        Ycoeff_v2 = np.array([320.5, 3.460])
        self.__m1m2_coeff_v2 = (Xcoeff_v2, Ycoeff_v2)

        return self

    def setTime(self, time: Frame):
        """
        Set parameters at given epoch
        :param tt_mjd: tdt time in MJD format
        :param xp: polar motion parameters in [radian]
        :param yp: [radian]
        :return:
        """
        tt_mjd = time.getTT_mjd
        if self.__externalEOP is None:
            xp, yp = time.getEOP['xp'], time.getEOP['yp']
        else:
            xp, yp = self.__externalEOP
        if self.__version == 1:
            self.__m1, self.__m2 = self.__polar2wobble_v1(tt_mjd, xp, yp)
        else:
            self.__m1, self.__m2 = self.__polar2wobble_v2(tt_mjd, xp, yp)

        return self

    def __getCS_solid(self):
        """
        It only works on C21 and S21, see Eq. 6.22
        :return:
        """
        Cpot, Spot = np.zeros(6), np.zeros(6)
        as2r = GeoMathKit.as2rad()

        #  radius to arc second
        mm1 = self.__m1 / as2r
        mm2 = self.__m2 / as2r

        Cpot[4] = -1.333e-9 * (mm1 + 0.0115 * mm2)
        Spot[4] = -1.333e-9 * (mm2 - 0.0115 * mm1)

        return Cpot, Spot

    def __getCS_ocean(self):
        """
        Desai (2002) presents a self-consistent equilibrium model of the ocean pole tide. This model accounts for
        continental boundaries, mass conservation over the oceans, self-gravitation, and loading of the ocean floor.
        Two formats are provided:
        1. (See Eq. 6.24)
        CS21 dominates the 90% of the total signal, which means extracting only the CS21 term can simplify the process
        2. (See Eq. 6.23)
        This is a more complex format, but can provide all components of ocean pole tide at any d/o less than 360
        :return: CS coefficients in one-dimension format
        """

        if self.__simpleOcean:
            Cpot, Spot = np.zeros(6), np.zeros(6)
            as2r = GeoMathKit.as2rad()

            #  radius to arc second
            mm1 = self.__m1 / as2r
            mm2 = self.__m2 / as2r

            Cpot[4] = -2.1778e-10 * (mm1 - 0.01724 * mm2)
            Spot[4] = -1.7232e-10 * (mm2 - 0.03365 * mm1)

            return Cpot, Spot

        Nmax = self.__maxdegree
        Cpot, Spot = np.zeros(int((Nmax + 2) * (Nmax + 1) / 2)), np.zeros(int((Nmax + 2) * (Nmax + 1) / 2))

        Cpot[1:] = self.__Rn * (self.__AnmR * self.__m1 * self.__RGAMA +
                                self.__AnmR * self.__m2 * self.__IGAMA +
                                self.__AnmI * self.__m2 * self.__RGAMA -
                                self.__AnmI * self.__m1 * self.__IGAMA)
        Spot[1:] = self.__Rn * (self.__BnmR * self.__m1 * self.__RGAMA +
                                self.__BnmR * self.__m2 * self.__IGAMA +
                                self.__BnmI * self.__m2 * self.__RGAMA -
                                self.__BnmI * self.__m1 * self.__IGAMA)

        return Cpot, Spot

    def getCS(self):
        """
        Get CS from solid only, ocean only or the sum of them.
        :param kind:
        :return:
        """
        if self.__kind == ['Solid Earth']:
            return self.__getCS_solid()
        elif self.__kind == ['Ocean']:
            return self.__getCS_ocean()
        elif set(self.__kind) == {'Solid Earth', 'Ocean'}:
            C1, S1 = self.__getCS_solid()
            C2, S2 = self.__getCS_ocean()
            return GeoMathKit.arrayAdd(C1, C2), GeoMathKit.arrayAdd(S1, S2)

    def __polar2wobble_v1(self, tt_mjd, xp, yp):
        """
        Transform the polar motion parameters to wobble parameters, see IERS2010 V1.0.0 section 7.1.4.
        Take care! This function waits to be updated!
        :param tt_mjd: tt time in MJD format
        :param xp: [radian]
        :param yp: [radian]
        :return: m1, m2 in [radian]
        """

        as2r = GeoMathKit.as2rad()

        '''Note that the original data used to generate the linear model used Besselian epochs. Thus, 
        strictly speaking, the time argument t in (7.25) is also a Besselian epoch. However, for all practical 
        purposes, a Julian epoch may be used for t. 

        '''
        # As a difference, MJD time (Dt) is fine as well.
        # 51544.5 denotes 2000.0
        J2000 = 51544.5
        # 2010.01.01
        J2010 = 55197
        Dt = (tt_mjd - J2000) / 365.25

        Dt_sequence = np.array([1, Dt, Dt ** 2, Dt ** 3])
        (Xcoeff_bf2010, Ycoeff_bf2010, Xcoeff_af2010, Ycoeff_af2010) = self.__m1m2_coeff_v1

        if tt_mjd <= J2010:
            Xcoeff = Xcoeff_bf2010
            Ycoeff = Ycoeff_bf2010
        else:
            Xcoeff = Xcoeff_af2010
            Ycoeff = Ycoeff_af2010

        # 3rd order polynomial
        xp_mean = np.sum(Xcoeff * Dt_sequence)
        yp_mean = np.sum(Ycoeff * Dt_sequence)

        # mas => radian
        m1 = xp - xp_mean / 1000 * as2r
        m2 = -(yp - yp_mean / 1000 * as2r)

        return m1, m2

    def __polar2wobble_v2(self, tt_mjd, xp, yp):
        """
        Transform the polar motion parameters to wobble parameters, see IERS2010 V1.2.0 section 7.1.4.
        Notice: Indeed, it only works from year 1900 to 2017
        :param tt_mjd: tt time in MJD format
        :param xp: [radian]
        :param yp: [radian]
        :return: m1, m2 in [radian]
        """

        as2r = GeoMathKit.as2rad()

        '''Note that the original data used to generate the linear model used Besselian epochs. Thus, 
        strictly speaking, the time argument t in (7.25) is also a Besselian epoch. However, for all practical 
        purposes, a Julian epoch may be used for t. 

        '''
        # As a difference, MJD time (Dt) is fine as well.
        # 51544.5 denotes 2000.0
        J2000 = 51544.5
        Dt = (tt_mjd - J2000) / 365.25

        Dt_sequence = np.array([1, Dt])
        (Xcoeff_v2, Ycoeff_v2) = self.__m1m2_coeff_v2

        # linear modelling
        xp_mean = np.sum(Xcoeff_v2 * Dt_sequence)
        yp_mean = np.sum(Ycoeff_v2 * Dt_sequence)

        # mas => radian
        m1 = xp - xp_mean / 1000 * as2r
        m2 = -(yp - yp_mean / 1000 * as2r)

        return m1, m2

    def setEOP(self, externalEOP):
        self.__externalEOP = externalEOP
        return self