import numpy as np
import warnings
from src.Preference.EnumType import TidesType, AODtype
import os


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
        self._FMConfig = None
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


class LoadGif48(LoadSH):
    """
    specified to read gif48 gravity fields. Plus, the file that formats the same as GIF48 can be read as well.
    """

    def __init__(self):
        LoadSH.__init__(self)
        # self.__sigmaC, self.__sigmaS = None, None
        # self.__C, self.__S = None, None

    def load(self, fileIn):
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

        stempC = np.zeros((int(self.maxDegree + 1), int(self.maxDegree + 1)))
        stempS = np.zeros((int(self.maxDegree + 1), int(self.maxDegree + 1)))
        stempsigmaC = np.zeros((int(self.maxDegree + 1), int(self.maxDegree + 1)))
        stempsigmaS = np.zeros((int(self.maxDegree + 1), int(self.maxDegree + 1)))
        for item in content:
            value = item.split()
            if len(value) == 0: continue
            stempC[int(value[1]), int(value[2])] = float(value[3].replace('D', 'E'))
            stempS[int(value[1]), int(value[2])] = float(value[4].replace('D', 'E'))
            stempsigmaC[int(value[1]), int(value[2])] = float(value[5].replace('D', 'E'))
            stempsigmaS[int(value[1]), int(value[2])] = float(value[6].replace('D', 'E'))

        for i in range(int(self.maxDegree + 1)):
            for j in range(i + 1):
                l.append(i)
                m.append(j)
                C.append(stempC[i, j])
                S.append(stempS[i, j])
                sigmaC.append(stempsigmaC[i, j])
                sigmaS.append(stempsigmaS[i, j])

        # for item in content:
        #     value = item.split()
        #     if len(value) == 0: continue
        #
        #     l.append(value[1])
        #     m.append(value[2])
        #     C.append(value[3])
        #     S.append(value[4])
        #     sigmaC.append(value[5])
        #     sigmaS.append(value[6])

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)
        sigmaC = np.array(sigmaC).astype(np.float64)
        sigmaS = np.array(sigmaS).astype(np.float64)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S
        self._sigmaC[(l * (l + 1) / 2 + m).astype(np.int64)] = sigmaC
        self._sigmaS[(l * (l + 1) / 2 + m).astype(np.int64)] = sigmaS


class LoadAtmosTide(LoadSH):
    """
    This class is used to read stokes coefficients at given epoch from AOD product.
    """

    def __init__(self):
        LoadSH.__init__(self)
        self.__AODdir = None

    def load(self, fileIn):
        """
        set the directory of AOD products to be read
        :param fileIn:
        :return:
        """
        self.__AODdir = fileIn
        return self

    def setInfo(self, tide: TidesType, kind, sincos='sin'):
        assert kind == AODtype.ATM.name or kind == AODtype.OCN.name
        assert sincos in ['sin', 'cos']

        '''search the directory by above information'''
        root_path = self.__AODdir
        target = None

        class Break(Exception):
            pass
        try:
            for root, dirs, files in os.walk(root_path):
                for name in files:
                    if (kind in name) and (tide.name in name):
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

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S


class LoadNonTide(LoadSH):
    """
    This class is used to read stokes coefficients at given epoch from AOD product.
    """

    def __init__(self):
        LoadSH.__init__(self)
        self.__kind = None
        self.__AODdir = None
        self.__box = {AODtype.ATM.name: 'atm',
                      AODtype.OCN.name: 'ocn',
                      AODtype.OBA.name: 'oba',
                      AODtype.GLO.name: 'glo'}
        self.__epoch = None

    def load(self, fileIn):
        """
        set the directory of AOD products to be read
        :param fileIn:
        :return:
        """
        self.__AODdir = fileIn
        return self

    def setTime(self, date: str, epoch: str):
        """
        specify the epoch of AOD file to be read
        :param date: eg., '2009-01-01'
        :param epoch: eg., '06:00:00'
        :return:
        """
        self.__epoch = epoch
        '''search the directory by month information'''
        root_path = self.__AODdir
        target = None
        class Break(Exception):
            pass

        try:
            for root, dirs, files in os.walk(root_path):
                for name in files:
                    if date in name:
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

        flag = self.__header(content)

        self.__read(content[flag + 1:])

        '''search the file by day information'''

        return self

    def setType(self, kind: str):
        """
        set the type to be read: ATM, OCN, GLO, OBA
        :return:
        """
        self.__kind = kind
        return self

    def __header(self, content: list):
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
            elif (self.__box[self.__kind] in content[i]) and (self.__epoch in content[i]):
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

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S


class LoadCSR(LoadSH):

    def __init__(self):
        LoadSH.__init__(self)
        # self.__sigmaC, self.__sigmaS = None, None
        # self.__C, self.__S = None, None

    def load(self, fileIn):
        """
        load CSR fields
        :param fileIn: CSR file and its path
        :return:
        """

        flag = 0

        with open(fileIn) as f:
            content = f.readlines()
            pass

        flag = self.__header_csr(content)

        self.__read_csr(content[flag + 1:])

        return self

    def __header_csr(self, content: list):
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

    def __read_csr(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S    sigma C    sigma S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        stempC = np.zeros((int(self.maxDegree + 1), int(self.maxDegree + 1)))
        stempS = np.zeros((int(self.maxDegree + 1), int(self.maxDegree + 1)))
        stempsigmaC = np.zeros((int(self.maxDegree + 1), int(self.maxDegree + 1)))
        stempsigmaS = np.zeros((int(self.maxDegree + 1), int(self.maxDegree + 1)))
        for item in content:
            value = item.split()
            if len(value) == 0: continue
            stempC[int(value[1]), int(value[2])] = float(value[3])
            stempS[int(value[1]), int(value[2])] = float(value[4])
            stempsigmaC[int(value[1]), int(value[2])] = float(value[5])
            stempsigmaS[int(value[1]), int(value[2])] = float(value[6])

        for i in range(int(self.maxDegree + 1)):
            for j in range(i + 1):
                l.append(i)
                m.append(j)
                C.append(stempC[i, j])
                S.append(stempS[i, j])
                sigmaC.append(stempsigmaC[i, j])
                sigmaS.append(stempsigmaS[i, j])

        # for item in content:
        #     value = item.split()
        #     if len(value) == 0: continue
        #
        #     l.append(value[1])
        #     m.append(value[2])
        #     C.append(value[3])
        #     S.append(value[4])
        #     sigmaC.append(value[5])
        #     sigmaS.append(value[6])

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)
        sigmaC = np.array(sigmaC).astype(np.float64)
        sigmaS = np.array(sigmaS).astype(np.float64)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S
        self._sigmaC[(l * (l + 1) / 2 + m).astype(np.int64)] = sigmaC
        self._sigmaS[(l * (l + 1) / 2 + m).astype(np.int64)] = sigmaS


class LoadGLO(LoadSH):
    """
    This class is used to read stokes coefficients at given epoch from Bai jiahui.
    """

    def __init__(self):
        LoadSH.__init__(self)
        self.__kind = None
        self.__AODdir = None
        self.__box = {AODtype.ATM.name: 'atm',
                      AODtype.OCN.name: 'ocn',
                      AODtype.OBA.name: 'oba',
                      AODtype.GLO.name: 'glo'}
        self.__epoch = None

    def load(self, fileIn):
        """
        set the directory of AOD products to be read
        :param fileIn:
        :return:
        """
        self.__AODdir = fileIn
        return self

    def setTime(self, date: str, epoch: str):
        """
        specify the epoch of AOD file to be read
        :param date: eg., '2009-01-01'
        :param epoch: eg., '06:00:00'
        :return:
        """
        self.__epoch = epoch
        '''search the directory by month information'''
        root_path = self.__AODdir
        target = None
        class Break(Exception):
            pass

        try:
            for root, dirs, files in os.walk(root_path):
                for name in files:
                    if date in name:
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

        flag = self.__header(content)

        self.__read(content[flag + 1:])

        '''search the file by day information'''

        return self

    def setType(self, kind: str):
        """
        set the type to be read: ATM, OCN, GLO, OBA
        :return:
        """
        self.__kind = kind
        return self

    def __header(self, content: list):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0
        self.maxDegree = int(180)

        for i in range(len(content)):
            value = content[i].split(':')
            if len(value) == 0:
                continue
            value[0] = value[0].strip()

            try:
                value[1] = value[1].replace('\n', '')
            except Exception as e:
                pass
            if (self.__box[self.__kind] in content[i]) and (self.__epoch in content[i]):
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

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S