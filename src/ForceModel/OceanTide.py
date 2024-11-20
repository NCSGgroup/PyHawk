from src.ForceModel.BaseGravCS import BaseGravCS
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Auxilary.FunNutArg import FunNutArg
from src.Preference.EnumType import InstanceOceanTide
from src.Preference.Pre_ForceModel import ForceModelConfig
import numpy as np


class AbsOceanTide(BaseGravCS):
    def __init__(self):
        super(AbsOceanTide, self).__init__()
        self._SR = None
        self._tdb_mjd = -1000000
        self._FMConfig = None
        self._OceanTideConfig = None
        self._PathOfFilesConfig = None
        self._ng = FunNutArg()

    def configure(self, FMConfig: ForceModelConfig):
        self._FMConfig = FMConfig
        self._OceanTideConfig = self._FMConfig.OceanTide()
        self._OceanTideConfig.__dict__.update(self._FMConfig.OceanTideConfig.copy())
        self._PathOfFilesConfig = self._FMConfig.PathOfFiles()
        self._PathOfFilesConfig.__dict__.update(self._FMConfig.PathOfFilesConfig.copy())
        return self

    def speedUp(self):
        self._SR = self._OceanTideConfig.gainFactor
        return self

    @staticmethod
    def new_OceanTide(kind:InstanceOceanTide):
        cls = None
        if kind is InstanceOceanTide.EOT11a:
            cls = EOT11a
        elif kind is InstanceOceanTide.FES2014:
            cls = FES2014
        inst = cls()
        return inst


class EOT11a(AbsOceanTide):
    """
    Notice: the input should be TDB time in MJD format.
    This is an EOT11a version modified from Matlab codes of Mayer Gurr
    """

    def __init__(self):
        super(EOT11a, self).__init__()
        self.__darwin = ('OM1', 'OM2', 'SA', 'SSA', 'MM', 'MF', 'MTM',
                         'MSQ', 'Q1', 'O1', 'P1', 'K1', '2N2', 'N2',
                         'M2', 'S2', 'K2', 'M4')
        self.__doodsonMatrix = np.zeros((256, 6), int)

    def load(self):
        """
        load the relevant files
        EXAMPLE: load("eot11a/")
        :param maxDegree: max degree of user-desired SH coefficients, notice that the maxDegree should be lower than 120
        :param fileDir: the path and dir of "eot11a"
        :return:
        """
        maxDegree = self._OceanTideConfig.maxDegree
        fileDir = self._PathOfFilesConfig.EOT11a
        assert maxDegree <= 120

        # read Stokes Coefficients from 18 major tide constitution
        self.__CnmCos, self.__SnmCos, self.__CnmSin, self.__SnmSin = [], [], [], []

        for i in range(len(self.__darwin)):
            c, s, GM, R = self.__read_gfc(fileDir + '/eot11a.' + self.__darwin[i] + '.cos.gfc')
            c, s = GeoMathKit.CS_2dTo1d(c[0:maxDegree + 1, 0:maxDegree + 1]), \
                   GeoMathKit.CS_2dTo1d(s[0:maxDegree + 1, 0:maxDegree + 1])
            self.__CnmCos.append(c), self.__SnmCos.append(s)

            c, s, GM, R = self.__read_gfc(fileDir + '/eot11a.' + self.__darwin[i] + '.sin.gfc')
            c, s = GeoMathKit.CS_2dTo1d(c[0:maxDegree + 1, 0:maxDegree + 1]), \
                   GeoMathKit.CS_2dTo1d(s[0:maxDegree + 1, 0:maxDegree + 1])
            self.__CnmSin.append(c), self.__SnmSin.append(s)

        self.__CnmCos, self.__SnmCos, self.__CnmSin, self.__SnmSin = np.array(self.__CnmCos), np.array(self.__SnmCos), \
                                                                     np.array(self.__CnmSin), np.array(self.__SnmSin)

        # read admittance: interpolation matrix (256 tides -> 18 major tides)
        self.__admittance = np.loadtxt(fileDir + '/admittance.linear.txt')

        # 256x6 matrix with doodson multipliers
        # 256 tides
        doodson = (
            '055.565', '055.575', '056.554', '056.556', '057.355', '057.553', '057.555', '057.565', '057.575',
            '058.554', '059.553', '062.656', '063.645', '063.655', '063.665', '064.456', '064.555', '065.445',
            '065.455', '065.465', '065.655', '065.665', '065.675', '066.454', '067.455', '067.465', '071.755',
            '072.556', '073.545', '073.555', '073.565', '074.556', '075.345', '075.355', '075.365', '075.555',
            '075.565', '075.575', '076.554', '077.355', '077.365', '081.655', '082.656', '083.445', '083.455',
            '083.655', '083.665', '083.675', '084.456', '085.255', '085.455', '085.465', '085.475', '086.454',
            '091.555', '092.556', '093.355', '093.555', '093.565', '093.575', '095.355', '095.365', '107.755',
            '109.555', '115.845', '115.855', '117.645', '117.655', '118.654', '119.455', '125.745', '125.755',
            '126.556', '126.754', '127.545', '127.555', '128.554', '129.355', '133.855', '134.656', '135.435',
            '135.635', '135.645', '135.655', '135.855', '136.555', '136.654', '137.445', '137.455', '137.655',
            '137.665', '138.454', '139.455', '143.535', '143.745', '143.755', '144.546', '144.556', '145.535',
            '145.545', '145.555', '145.755', '145.765', '146.554', '147.355', '147.555', '147.565', '148.554',
            '153.645', '153.655', '154.656', '155.435', '155.445', '155.455', '155.645', '155.655', '155.665',
            '155.675', '156.555', '156.654', '157.445', '157.455', '157.465', '158.454', '161.557', '162.556',
            '163.545', '163.555', '163.755', '164.554', '164.556', '165.545', '165.555', '165.565', '165.575',
            '166.554', '167.355', '167.555', '167.565', '168.554', '172.656', '173.445', '173.645', '173.655',
            '173.665', '174.456', '174.555', '175.445', '175.455', '175.465', '175.655', '175.665', '175.675',
            '176.454', '182.556', '183.545', '183.555', '183.565', '185.355', '185.365', '185.555', '185.565',
            '185.575', '191.655', '193.455', '193.465', '193.655', '193.665', '195.255', '195.455', '195.465',
            '195.475', '207.855', '209.655', '215.955', '217.755', '219.555', '225.855', '227.645', '227.655',
            '228.654', '229.455', '234.756', '235.745', '235.755', '236.556', '236.655', '236.754', '237.545',
            '237.555', '238.554', '239.355', '243.635', '243.855', '244.656', '245.435', '245.645', '245.655',
            '246.456', '246.555', '246.654', '247.445', '247.455', '247.655', '248.454', '253.535', '253.755',
            '254.556', '255.535', '255.545', '255.555', '255.557', '255.755', '255.765', '256.554', '257.355',
            '257.555', '257.565', '257.575', '262.656', '263.645', '263.655', '264.555', '265.445', '265.455',
            '265.655', '265.665', '265.675', '267.455', '267.465', '271.557', '272.556', '273.545', '273.555',
            '273.557', '274.554', '274.556', '275.545', '275.555', '275.565', '275.575', '276.554', '277.555',
            '283.655', '283.665', '285.455', '285.465', '285.475', '293.555', '293.565', '295.355', '295.365',
            '295.555', '295.565', '295.575', '455.555')

        for i in range(len(doodson)):
            self.__doodsonMatrix[i][0] = int(doodson[i][0]) - 0
            self.__doodsonMatrix[i][1] = int(doodson[i][1]) - 5
            self.__doodsonMatrix[i][2] = int(doodson[i][2]) - 5
            self.__doodsonMatrix[i][3] = int(doodson[i][4]) - 5
            self.__doodsonMatrix[i][4] = int(doodson[i][5]) - 5
            self.__doodsonMatrix[i][5] = int(doodson[i][6]) - 5

        return self

    def __read_gfc(self, filename):
        """
        Read tide gfc files in order
        :param filename: filename of each tide constitution
        :return:
        """

        with open(filename) as f:
            content = f.readlines()

        GM = float(content[10].split()[1])
        R = float(content[11].split()[1])
        N = int(content[12].split()[1])

        C, S = np.zeros((N + 1, N + 1)), np.zeros((N + 1, N + 1))

        for string in content[18:]:
            val = string.split()
            C[int(val[1]), int(val[2])] = float(val[3])
            S[int(val[1]), int(val[2])] = float(val[4])

        return C, S, GM, R

    def getCS(self):
        """
        compute Stokes coefficients at given epoch
        :param tdb_mjd: given TDB time in MJD format
        :param theta: CORRESPONDING MEAN SID.TIME GREENWICH. = GMST
        :return: one-dimensional SH coefficients up to max degree defined by user
        """
        tdb_mjd = self._time.getTDB_mjd
        theta = self._time.getTheta
        if np.fabs(tdb_mjd - self._tdb_mjd) < (self._SR / 86400.0):
            '''To avoid too dense output and enhance computation efficiency'''
            return self._c.copy(), self._s.copy()

        self._tdb_mjd = tdb_mjd

        ds = self._ng.getNutArg(tdb_mjd, theta)

        # 256 tides's thetaS
        thetaf = np.matmul(self.__doodsonMatrix, ds)

        # thetaf = np.matmul(self.__doodsonMatrix, GeoMathKit.nutarg(tdb_mjd, theta))

        factorcos = np.matmul(self.__admittance, np.cos(thetaf))

        factorsin = np.matmul(self.__admittance, np.sin(thetaf))

        Cnm = factorcos * self.__CnmCos + factorsin * self.__CnmSin
        Snm = factorcos * self.__SnmCos + factorsin * self.__SnmSin

        self._c, self._s = Cnm.sum(0), Snm.sum(0)

        return self._c.copy(), self._s.copy()


class FES2014(AbsOceanTide):

    def __init__(self):
        super(FES2014, self).__init__()
        self.__darwin = ('om1', 'om2', 'sa', 'ssa', 'mm', 'mf', 'mtm', 'msq', 'q1', 'o1',
                         'p1', 's1', 'k1', 'j1', 'eps2', '2n2', 'mu2', 'n2', 'nu2', 'm2', 'la2',
                         'l2', 't2', 's2', 'r2', 'k2', 'm3', 'n4', 'mn4', 'm4', 'ms4', 's4', 'm6', 'm8')
        # 361 doodsons
        self.__doodsonMatrix = np.zeros((361, 6), int)

    def load(self):
        """
        load the relevant files
        EXAMPLE: load("eot11a/")
        :param maxDegree: max degree of user-desired SH coefficients, notice that the maxDegree should be lower than 120
        :param fileDir: the path and dir of "eot11a"
        :return:
        """
        maxDegree = self._OceanTideConfig.maxDegree
        fileDir = self._PathOfFilesConfig.FES2014
        assert maxDegree <= 180

        # read Stokes Coefficients from 18 major tide constitution
        self.__CnmCos, self.__SnmCos, self.__CnmSin, self.__SnmSin = [], [], [], []

        for i in range(len(self.__darwin)):
            c, s, GM, R = self.__read_gfc(fileDir + '/fes2014b_n180_version20170520.'
                                                           + self.__darwin[i] + '.cos.gfc')
            c, s = GeoMathKit.CS_2dTo1d(c[0:maxDegree + 1, 0:maxDegree + 1]), \
                   GeoMathKit.CS_2dTo1d(s[0:maxDegree + 1, 0:maxDegree + 1])
            self.__CnmCos.append(c), self.__SnmCos.append(s)

            c, s, GM, R = self.__read_gfc(fileDir + '/fes2014b_n180_version20170520.'
                                                           + self.__darwin[i] + '.sin.gfc')
            c, s = GeoMathKit.CS_2dTo1d(c[0:maxDegree + 1, 0:maxDegree + 1]), \
                   GeoMathKit.CS_2dTo1d(s[0:maxDegree + 1, 0:maxDegree + 1])
            self.__CnmSin.append(c), self.__SnmSin.append(s)

        self.__CnmCos, self.__SnmCos, self.__CnmSin, self.__SnmSin = np.array(self.__CnmCos), np.array(self.__SnmCos), \
                                                                     np.array(self.__CnmSin), np.array(self.__SnmSin)

        # read admittance: interpolation matrix (361 tides -> 18 major tides)
        self.__admittance = np.loadtxt(fileDir + '/fes2014b_admittance_linear_linear.txt')

        # 361x6 matrix with doodson multipliers
        # 361 tides
        doodson = np.loadtxt(fileDir + '/admittanceTides.txt', comments='#', dtype=str)

        for i in range(len(doodson)):
            self.__doodsonMatrix[i][0] = int(doodson[i][0], base=16) - 0
            self.__doodsonMatrix[i][1] = int(doodson[i][1], base=16) - 5
            self.__doodsonMatrix[i][2] = int(doodson[i][2], base=16) - 5
            self.__doodsonMatrix[i][3] = int(doodson[i][4], base=16) - 5
            self.__doodsonMatrix[i][4] = int(doodson[i][5], base=16) - 5
            self.__doodsonMatrix[i][5] = int(doodson[i][6], base=16) - 5

        return self

    def __read_gfc(self, filename):
        """
        Read tide gfc files in order
        :param filename: filename of each tide constitution
        :return:
        """

        with open(filename) as f:
            content = f.readlines()

        GM = float(content[13].split()[1])
        R = float(content[14].split()[1])
        N = int(content[15].split()[1])

        C, S = np.zeros((N + 1, N + 1)), np.zeros((N + 1, N + 1))

        for string in content[21:]:
            val = string.split()
            C[int(val[1]), int(val[2])] = float(val[3])
            S[int(val[1]), int(val[2])] = float(val[4])

        return C, S, GM, R

    def getCS(self):
        """
        compute Stokes coefficients at given epoch
        :param tdb_mjd: given TDB time in MJD format
        :param theta: CORRESPONDING MEAN SID.TIME GREENWICH. = GMST
        :return: one-dimensional SH coefficients up to max degree defined by user
        """
        tdb_mjd = self._time.getTDB_mjd
        theta = self._time.getTheta
        if np.fabs(tdb_mjd - self._tdb_mjd) < (self._SR / 86400.0):
            '''To avoid too dense output and enhance computation efficiency'''
            return self._c.copy(), self._s.copy()

        self._tdb_mjd = tdb_mjd

        ds = self._ng.getNutArg(tdb_mjd, theta)
        thetaf = np.matmul(self.__doodsonMatrix, ds)
        # thetaf = np.matmul(self.__doodsonMatrix, GeoMathKit.nutarg(tdb_mjd, theta))

        factorcos = np.matmul(self.__admittance, np.cos(thetaf))

        factorsin = np.matmul(self.__admittance, np.sin(thetaf))

        Cnm = factorcos * self.__CnmCos + factorsin * self.__CnmSin
        Snm = factorcos * self.__SnmCos + factorsin * self.__SnmSin
        self._c, self._s = Cnm.sum(0), Snm.sum(0)

        return self._c.copy(), self._s.copy()
