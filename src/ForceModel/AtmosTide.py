from src.ForceModel.BaseGravCS import BaseGravCS
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Auxilary.FunNutArg import FunNutArg
from src.ForceModel.LoadSH import LoadSH
from src.Preference.Pre_ForceModel import AtmosTideConfig, PathOfFiles
from src.Preference.EnumType import TidesType, AODtype
import numpy as np
import os


class AbsAtmosTide(BaseGravCS):

    def __init__(self):
        super(AbsAtmosTide, self).__init__()
        self._C = None
        self._S = None
        self._SR = None
        self._tdb_mjd = -1000000
        self.setSampleRate()
        self._ng = FunNutArg()

    def setSampleRate(self, SR=AtmosTideConfig.SR):
        """
        To decide the output's maximal sampling rate
        :param SR: the desired sample rate [seconds]
        :return:
        """
        self._SR = SR
        return self

    def getCS(self):
        """
        compute Stokes coefficients at given epoch
        :param tdb_mjd: given TDB time in MJD format
        :param theta: CORRESPONDING MEAN SID.TIME GREENWICH. = GMST
        :return: one-dimensional SH coefficients up to max degree defined by user
        """
        return self._C.copy(), self._S.copy()


class AtmosBB2003(AbsAtmosTide):

    def __init__(self, kind=AtmosTideConfig.BB2003Kind):
        super(AtmosBB2003, self).__init__()
        self.__load_BB2003()
        self.__kind = [key for key, value in kind.items() if value]
        self.__nutarg = None

    def getCS(self):
        """
        get CS by defining the tide constituents.
        Sa is unavailable now.
        :param tdb_mjd: tdb in MJD format, set parameters and compute nutation arguments
        :param theta: mean GST
        :return:
        """
        tdb_mjd = self._time.getTDB_mjd
        theta = self._time.getTheta
        self.__nutarg = self._ng.getNutArg(tdb_mjd, theta)

        C, S = np.zeros(45), np.zeros(45)
        for term in self.__kind:
            C0, S0 = self.__getTide(term)
            C = GeoMathKit.arrayAdd(C0, C)
            S = GeoMathKit.arrayAdd(S0, S)

        self._C, self._S = C, S
        return self._C.copy(), self._S.copy()

    def __load_BB2003(self):
        """
        Only S1 and S2 are offered here
        Notice: the unit is given in [mbar] !  1 mbar = 100 Pa  => 1cm EWH
        :return:
        """
        self.__ATM = {'S1': None, 'S2': None, 'Sa': None, 'Ssa': None}
        '''
        Format = 1:  Csin+, Ccos+, Csin-, Ccos-
        Format = 2: C+, eps+, C-, eps-
        '''

        S2 = np.array(
            [
                [0, 0, 0.005, 345.0],
                [1, 0, 0.003, 112.0],
                [2, 0, 0.120, 157.0],
                [3, 0, 0.012, 72.0],
                [1, 1, 0.025, 96.9],
                [2, 1, 0.025, 152.8],
                [3, 1, 0.022, 296.0],
                [4, 1, 0.004, 285.4],
                [2, 2, 0.545, 148.8],
                [3, 2, 0.028, 20.2],
                [4, 2, 0.079, 332.8],
                [5, 2, 0.005, 236.9],
                [3, 3, 0.018, 71.8],
                [4, 3, 0.004, 91.3],
                [5, 3, 0.008, 322.8],
                [6, 3, 0.002, 49.9],
                [4, 4, 0.015, 273.1],
                [5, 4, 0.008, 16.6],
                [6, 4, 0.002, 285.4],
                [7, 4, 0.004, 256.8],
                [5, 5, 0.009, 94.9],
                [6, 5, 0.001, 30.1],
                [7, 5, 0.004, 90.8],
                [8, 5, 0.005, 74.9]
            ]
        )

        S1 = np.array(
            [
                [0, 0, 0.011, 295.1],
                [1, 0, 0.014, 301.7],
                [2, 0, 0.029, 228.6],
                [3, 0, 0.019, 266.8],
                [1, 1, 0.220, 73.2],
                [2, 1, 0.019, 37.6],
                [3, 1, 0.125, 264.8],
                [4, 1, 0.016, 155.8],
                [2, 2, 0.066, 37.7],
                [3, 2, 0.025, 358.1],
                [4, 2, 0.031, 241.7],
                [5, 2, 0.003, 147.4],
                [3, 3, 0.020, 235.6],
                [4, 3, 0.032, 298.3],
                [5, 3, 0.019, 258.3],
                [6, 3, 0.011, 186.3],
                [4, 4, 0.029, 336.3],
                [5, 4, 0.014, 100.1],
                [6, 4, 0.010, 34.8],
                [7, 4, 0.011, 344.7],
                [5, 5, 0.037, 327.4],
                [6, 5, 0.034, 93.0],
                [7, 5, 0.012, 83.9],
                [8, 5, 0.007, 266.1]
            ]
        )

        GM = 0.3986004415000e15
        AE = 0.6378136600000e07
        # average density of water
        RHOW = 0.1025000000000e4
        G = 6.6700000000000e-11

        Kn = np.array([
            0.000e0,
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

        '''
        Attribute terms by Format 1
        see Eq. 6 nd Eq. 8 at page 4 in reference 4
        '''
        wave = None

        for item in ('S1', 'S2'):
            if item == 'S1':
                wave = S1
            elif item == 'S2':
                wave = S2

            shape = np.shape(wave)
            Cplus, Splus, Cminus, Sminus = np.zeros(shape[0]), np.zeros(shape[0]), \
                                           np.zeros(shape[0]), np.zeros(shape[0])
            # prograde waves
            Cplus = wave[:, 2] * np.sin(np.deg2rad(wave[:, 3]))
            Splus = wave[:, 2] * np.cos(np.deg2rad(wave[:, 3]))

            # retrograde waves
            if shape[1] == 6:
                Cminus = wave[:, 4] * np.sin(np.deg2rad(wave[:, 5]))
                Sminus = wave[:, 4] * np.cos(np.deg2rad(wave[:, 5]))

            l, m = wave[:, 0].astype(np.int), wave[:, 1].astype(np.int)
            lm = l * (l + 1) / 2 + m
            lm = lm.astype(np.int)
            # 1 bar = 100 mbar, thus the factor should be divided by 100 to keep unitless
            factor = 4 * np.pi * G * AE ** 2 * RHOW / GM * (1 + Kn[l]) / (2 * l + 1) / 100

            self.__ATM[item] = (Cplus, Splus, Cminus, Sminus, lm, self.__doodsonMatrix(item), factor)

        pass

    def __getTide(self, tide='S1'):

        (Cplus, Splus, Cminus, Sminus, lm, doodsonMatrix, factor) = self.__ATM[tide]

        thetaf = np.matmul(doodsonMatrix, self.__nutarg)

        C, S = np.zeros(lm[-1] + 1), np.zeros(lm[-1] + 1)

        C[lm] = factor * ((Cplus + Cminus) * np.cos(thetaf) + (Splus + Sminus) * np.sin(thetaf))
        S[lm] = factor * ((Splus - Sminus) * np.cos(thetaf) - (Cplus - Cminus) * np.sin(thetaf))

        '''
        S[x0] should be equal to 0
        '''
        S = GeoMathKit.Sx0(S)
        return C, S

    def __doodsonMatrix(self, tide='S1'):

        DoodsonNumber = {'S1': '164.556', 'S2': '273.555', 'Sa': '056.554', 'Ssa': '057.555'}

        doodson = DoodsonNumber[tide]

        doodsonMatrix = np.zeros(6)

        doodsonMatrix[0] = int(doodson[0]) - 0
        doodsonMatrix[1] = int(doodson[1]) - 5
        doodsonMatrix[2] = int(doodson[2]) - 5
        doodsonMatrix[3] = int(doodson[4]) - 5
        doodsonMatrix[4] = int(doodson[5]) - 5
        doodsonMatrix[5] = int(doodson[6]) - 5

        return doodsonMatrix


class AtmosAOD(AbsAtmosTide):

    def __init__(self, kind=AtmosTideConfig.tide):
        super(AtmosAOD, self).__init__()
        self.__kind = [key.name for key, value in kind.items() if value]

        doodson = {
            'S1': '164.555',
            'S2': '273.555',
            'M2': '255.555',
            'S3': '382.555',
            'P1': '163.555',
            'K1': '165.555',
            'N2': '245.655',
            'L2': '265.455',
            'T2': '272.556',
            'R2': '274.554',
            'T3': '381.555',
            'R3': '383.555'
        }

        '''construct the doodson matrix for the given tide'''
        doodsonMatrix = {}
        for tide in self.__kind:
            doodsonMatrix[tide] = np.array([int(doodson[tide][0]) - 0,
                                            int(doodson[tide][1]) - 5,
                                            int(doodson[tide][2]) - 5,
                                            int(doodson[tide][4]) - 5,
                                            int(doodson[tide][5]) - 5,
                                            int(doodson[tide][6]) - 5])

        self.__doodsonMatrix = np.array([doodsonMatrix[k] for k in self.__kind])

        '''warburg correction: Notice, this differs from the AOD document for P1 and K1 term'''
        warburg = {
            'S1': np.pi,
            'S2': 0,
            'M2': 0,
            'S3': 0,

            'P1': np.pi / 2,
            'K1': -np.pi / 2,

            'N2': 0,
            'L2': np.pi,
            'T2': 0,
            'R2': np.pi,
            'T3': 0,
            'R3': 0
        }

        self.__warburg = np.array([warburg[k] for k in self.__kind])

        self.__tideCS = None

    def loadfile(self, fileDir=PathOfFiles.Atmos, Nmax: int=AtmosTideConfig.Nmax):
        """

        :param Nmax: max degree to be extracted
        :param fileDir: path of the tide files
        :return:
        """

        adt = AODtides().load(fileIn=fileDir)
        spec = AODtype.ATM

        tideCS = {}
        for tide in self.__kind:
            cnmSin, snmSin = adt.setInfo(tide=TidesType[tide], kind=spec, sincos='sin').getCS(Nmax)
            cnmCos, snmCos = adt.setInfo(tide=TidesType[tide], kind=spec, sincos='cos').getCS(Nmax)
            tideCS[tide] = np.array([cnmSin, snmSin, cnmCos, snmCos])

        """sort"""
        self.__tideCS = np.array([tideCS[k] for k in self.__kind])

        return self

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
            return self._C.copy(), self._S.copy()

        self._tdb_mjd = tdb_mjd

        ds = self._ng.getNutArg(tdb_mjd, theta)
        thetaf = np.matmul(self.__doodsonMatrix, ds) +  self.__warburg.reshape((-1,1))
        cnmSin, snmSin, cnmCos, snmCos = self.__tideCS[:, 0, :], self.__tideCS[:, 1, :], \
                                         self.__tideCS[:, 2, :], self.__tideCS[:, 3, :]

        Cnm = cnmCos * np.cos(thetaf) + cnmSin * np.sin(thetaf)
        Snm = snmCos * np.cos(thetaf) + snmSin * np.sin(thetaf)
        self._C, self._S = Cnm.sum(0), Snm.sum(0)

        return self._C.copy(), self._S.copy()


class AODtides(LoadSH):
    """
    This class is used to read stokes coefficients at given epoch from AOD product.
    """

    def __init__(self):
        LoadSH.__init__(self)
        self.__AODdir = None

    def load(self, fileIn = PathOfFiles.Atmos):
        """
        set the directory of AOD products to be read
        :param fileIn:
        :return:
        """
        self.__AODdir = fileIn
        return self

    def setInfo(self, tide: TidesType, kind=AtmosTideConfig.kind, sincos='sin'):
        assert kind == AODtype.ATM or kind == AODtype.OCN
        assert sincos in ['sin', 'cos']

        '''search the directory by above information'''
        root_path = self.__AODdir
        target = None

        class Break(Exception):
            pass
        try:
            for root, dirs, files in os.walk(root_path):
                for name in files:
                    if (kind.name in name) and (tide.name in name):
                        target = os.path.join(root, name)
                        raise Break
        except Break as e:
            pass

        flag = 0

        if target is None:
            raise FileNotFoundError


        with open(target) as f:
            content = f.readlines()
            pass

        flag = self.__header(content, sincos)

        self.__read(content[flag + 1:])

        return self

    def __header(self, content: list, sincos: str):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0

        for i in range(len(content)):
            value = content[i].split(':')
            if len(value) == 0:
                continue
            value[0] = value[0].strip()
            try:
                value[1] = value[1].replace('\n', '')
            except Exception as e:
                pass

            if value[0] == 'SOFTWARE VERSION':
                self.product_type = value[1]
            elif value[0] == 'PRODUCER AGENCY':
                self.modelname = value[1]
            elif value[0] == 'CONSTANT GM [M^3/S^2]':
                self.GM = float(value[1])
            elif value[0] == 'CONSTANT A [M]':
                self.Radius = float(value[1])
            elif value[0] == 'MAXIMUM DEGREE':
                self.maxDegree = int(value[1])
            elif value[0] == 'COEFFICIENT ERRORS (YES/NO)':
                self.errors = value[1]
            elif value[0] == 'norm':
                self.norm = value[1]
            elif value[0] == 'tide_system':
                self.zero_tide = (value[1] == 'zero_tide')
            elif sincos in content[i]:
                flag = i
                break

        return flag

    def __read(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S    sigma C    sigma S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0:
                continue
            if len(item.split(':')) > 1:
                break

            l.append(value[0])
            m.append(value[1])
            C.append(value[2])
            S.append(value[3])

        l = np.array(l).astype(np.int)
        m = np.array(m).astype(np.int)
        C = np.array(C).astype(np.float)
        S = np.array(S).astype(np.float)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int)] = S

