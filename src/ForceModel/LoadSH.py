import numpy as np
import warnings
from src.Preference.Pre_ForceModel import PathOfFiles


class LoadSH:
    """
    This class is a base class that deals with the gravity models reading.
    """

    def __init__(self):
        self.product_type = None
        self.modelname = None
        self.GM = None
        self.Radius = None
        self.maxDegree = None
        self.zero_tide = None
        self.errors = None
        self.norm = None

        self._sigmaC, self._sigmaS = None, None
        self._C, self._S = None, None
        pass

    def load(self, fileIn):
        return self

    def getCS(self, Nmax):

        assert Nmax >= 0

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        if end > len(self._C):
            warnings.warn('Nmax is too big')

        return self._C[0:end].copy(), self._S[0:end].copy()

    def getSigmaCS(self, Nmax):

        assert Nmax >= 0

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        if end > len(self._C):
            warnings.warn('Nmax is too big')

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        return self._sigmaC[0:end], self._sigmaS[0:end]


class Gif48(LoadSH):
    """
    specified to read gif48 gravity fields. Plus, the file that formats the same as GIF48 can be read as well.
    """

    def __init__(self):
        LoadSH.__init__(self)
        # self.__sigmaC, self.__sigmaS = None, None
        # self.__C, self.__S = None, None

    def load(self, fileIn = PathOfFiles.Gif48):
        """
        load gif48 fields
        :param fileIn: gif48 file and its path
        :return:
        """

        flag = 0

        with open(fileIn) as f:
            content = f.readlines()
            pass

        flag = self.__header_gif48(content)

        self.__read_gif48(content[flag + 1:])

        return self

    def __header_gif48(self, content: list):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0

        for i in range(len(content)):
            value = content[i].split()
            if len(value) == 0: continue

            if value[0] == 'product_type':
                self.product_type = value[1]
            elif value[0] == 'modelname':
                self.modelname = value[1]
            elif value[0] == 'earth_gravity_constant':
                self.GM = float(value[1])
            elif value[0] == 'radius':
                self.Radius = float(value[1])
            elif value[0] == 'max_degree':
                self.maxDegree = int(value[1])
            elif value[0] == 'errors':
                self.errors = value[1]
            elif value[0] == 'norm':
                self.norm = value[1]
            elif value[0] == 'tide_system':
                self.zero_tide = (value[1] == 'zero_tide')
            elif value[0] == 'end_of_head':
                flag = i
                break

        return flag

    def __read_gif48(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S    sigma C    sigma S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0: continue

            l.append(value[1])
            m.append(value[2])
            C.append(value[3])
            S.append(value[4])
            sigmaC.append(value[5])
            sigmaS.append(value[6])

        l = np.array(l).astype(np.int)
        m = np.array(m).astype(np.int)
        C = np.array(C).astype(np.float)
        S = np.array(S).astype(np.float)
        sigmaC = np.array(sigmaC).astype(np.float)
        sigmaS = np.array(sigmaS).astype(np.float)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int)] = S
        self._sigmaC[(l * (l + 1) / 2 + m).astype(np.int)] = sigmaC
        self._sigmaS[(l * (l + 1) / 2 + m).astype(np.int)] = sigmaS

