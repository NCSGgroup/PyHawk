import numpy as np
import warnings
from src.Preference.EnumType import TidesType, AODtype
import os
from datetime import datetime
from pathlib import Path
from src.Auxilary.FunNutArg import FunNutArg


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


class LoadGOT56(LoadSH):
    """Read TU Graz GOT5.6_OCN potential coefficients, including ocean loading.

    load(gfc) reads one component in its native CE frame. load(directory) reads
    the complete atlas; then call setTime(tdb_mjd, theta) before getCS(Nmax).
    theta is GMST in radians from Frame. Synthesis defaults to CM (degree one
    zero). Do not add loading, nodal or phase corrections again.
    Coefficient index: n*(n+1)//2 + m.
    """

    _basename = 'GOT5.6_OCN'

    def __init__(self):
        super().__init__()
        self.fileList = None
        self.doodsonMatrix = None
        self.admittance = None
        self.sourceMaxDegree = None
        self.reference_frame = 'CE'
        self.output_frame = 'CE'
        self.includes_loading = True
        self._atlas = None

    @staticmethod
    def _degree(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('Degree must be a non-negative integer')
        return int(value)

    @staticmethod
    def _number(value):
        return float(value.replace('D', 'E').replace('d', 'e'))

    @classmethod
    def _read_gfc(cls, filename, maxDegree):
        header = {}
        with open(filename, encoding='utf-8-sig') as stream:
            for line in stream:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] in ('end_of_head', 'end_of_header'):
                    break
                if len(fields) > 1:
                    header[fields[0]] = fields[1]
            else:
                raise ValueError('Missing end_of_head in %s' % filename)
            required = ('modelname', 'product_type', 'earth_gravity_constant',
                        'radius', 'max_degree', 'norm', 'errors')
            if any(key not in header for key in required):
                raise ValueError('Incomplete GOT5.6 header in %s' % filename)
            if (not header['modelname'].startswith('GOT5.6') or
                    header['product_type'] != 'gravity_field' or
                    header['norm'] != 'fully_normalized' or header['errors'] != 'no'):
                raise ValueError('Unsupported GOT5.6 coefficient convention in %s' % filename)
            source_degree = cls._degree(int(header['max_degree']))
            degree = source_degree if maxDegree is None else cls._degree(maxDegree)
            if degree > source_degree:
                raise ValueError('Requested degree exceeds file max_degree in %s' % filename)
            size = (degree + 1) * (degree + 2) // 2
            c, s = np.zeros(size), np.zeros(size)
            seen = np.zeros((source_degree + 1) * (source_degree + 2) // 2, dtype=bool)
            for line in stream:
                fields = line.split()
                if not fields or fields[0].startswith(('#', '*')):
                    continue
                if fields[0] != 'gfc' or len(fields) != 5:
                    raise ValueError('Invalid GOT5.6 gfc record in %s' % filename)
                n, m = int(fields[1]), int(fields[2])
                if not 0 <= m <= n <= source_degree:
                    raise ValueError('Invalid degree/order in %s' % filename)
                index = n * (n + 1) // 2 + m
                if seen[index]:
                    raise ValueError('Duplicate coefficient in %s' % filename)
                seen[index] = True
                cv, sv = cls._number(fields[3]), cls._number(fields[4])
                if not np.isfinite([cv, sv]).all():
                    raise ValueError('Non-finite coefficient in %s' % filename)
                if n <= degree:
                    c[index], s[index] = cv, sv
            if not seen.all():
                raise ValueError('Incomplete coefficient triangle in %s' % filename)
        metadata = dict(product_type=header['product_type'], modelname=header['modelname'],
                        GM=cls._number(header['earth_gravity_constant']),
                        Radius=cls._number(header['radius']), maxDegree=degree,
                        sourceMaxDegree=source_degree, norm=header['norm'],
                        errors=header['errors'], zero_tide=None)
        if not np.isfinite([metadata['GM'], metadata['Radius']]).all() or min(metadata['GM'], metadata['Radius']) <= 0:
            raise ValueError('Invalid GM/radius in %s' % filename)
        return c, s, metadata

    def load(self, fileIn, maxDegree=None):
        """Load a GFC or directory with the official file list; None keeps degree 180."""
        path = Path(fileIn)
        if path.is_file():
            c, s, metadata = self._read_gfc(path, maxDegree)
            self.__init__()
            self.__dict__.update(metadata)
            self._C, self._S = c, s
            return self
        if not path.is_dir():
            raise FileNotFoundError(path)
        file_list = np.loadtxt(path / (self._basename + '_001fileList.txt'), dtype=str, ndmin=2)
        doodson = np.loadtxt(path / (self._basename + '_002doodson.txt'), ndmin=2)
        admittance = np.loadtxt(path / (self._basename + '_003admittance.txt'), ndmin=2)
        if file_list.shape[1] != 2 or file_list.shape[0] == 0:
            raise ValueError('GOT5.6 file list must contain cos/sin filename pairs')
        if (doodson.shape[1] != 6 or not np.isfinite(doodson).all() or
                not np.equal(doodson, np.floor(doodson)).all()):
            raise ValueError('Doodson matrix must contain six integer multipliers per row')
        if admittance.shape != (len(file_list), len(doodson)) or not np.isfinite(admittance).all():
            raise ValueError('Admittance dimensions must match atlas rows and Doodson rows')
        atlas = [[], [], [], []]  # Ccos, Scos, Csin, Ssin
        metadata = None
        for pair in file_list:
            for part, name in enumerate(pair):
                component = Path(name)
                if component.is_absolute() or len(component.parts) != 1 or component.suffix != '.gfc':
                    raise ValueError('Invalid component filename: %s' % name)
                c, s, current = self._read_gfc(path / component, maxDegree)
                if metadata is None:
                    metadata = current
                elif any(current[key] != metadata[key] for key in
                         ('GM', 'Radius', 'maxDegree', 'sourceMaxDegree', 'norm')):
                    raise ValueError('Inconsistent coefficient metadata: %s' % name)
                atlas[2 * part].append(c)
                atlas[2 * part + 1].append(s)
        self.__init__()
        self.__dict__.update(metadata)
        self.modelname = self._basename
        self.fileList = file_list.copy()
        self.doodsonMatrix = doodson.astype(np.int64)
        self.admittance = admittance
        self._atlas = tuple(np.array(part) for part in atlas)
        return self

    def setDoodsonArguments(self, beta, frame='CM'):
        """Synthesize from six Doodson angles in radians, without extra corrections."""
        if self._atlas is None:
            raise RuntimeError('Load the complete GOT5.6_OCN directory before synthesis')
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.shape != (6,) or not np.isfinite(beta).all():
            raise ValueError('Six finite Doodson angles are required')
        if frame not in ('CE', 'CM'):
            raise ValueError('frame must be CE or CM')
        phase = self.doodsonMatrix @ beta
        fc, fs = self.admittance @ np.cos(phase), self.admittance @ np.sin(phase)
        cc, sc, cs, ss = self._atlas
        # Match EOT11a's elementwise synthesis. Avoid launching a large
        # BLAS thread team for each short, wide coefficient contraction.
        self._C = (fc[:, None] * cc + fs[:, None] * cs).sum(axis=0)
        self._S = (fc[:, None] * sc + fs[:, None] * ss).sum(axis=0)
        if frame == 'CM':
            self._C[1:3] = 0.0
            self._S[1:3] = 0.0
        self.output_frame = frame
        return self

    def setTime(self, tdb_mjd, theta, frame='CM'):
        """Use the same TDB MJD and GMST (radians) as existing OceanTide models."""
        if not np.isfinite([tdb_mjd, theta]).all():
            raise ValueError('TDB MJD and GMST must be finite')
        beta = FunNutArg().getNutArg(tdb_mjd, theta)
        return self.setDoodsonArguments(beta, frame=frame)

    def getCS(self, Nmax=None):
        if self._C is None:
            raise RuntimeError('Load a component, or load an atlas and setTime before getCS')
        degree = self.maxDegree if Nmax is None else self._degree(Nmax)
        if degree > self.maxDegree:
            raise ValueError('Requested degree exceeds loaded GOT5.6 coefficients')
        return super().getCS(degree)

    def getSigmaCS(self, Nmax):
        raise NotImplementedError('GOT5.6_OCN declares errors=no; no formal uncertainties are supplied')


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


class LoadGOCO06S(LoadSH):
    """
    specified to read gif48 gravity fields. Plus, the file that formats the same as GIF48 can be read as well.
    """

    def __init__(self):
        LoadSH.__init__(self)
        # self.__sigmaC, self.__sigmaS = None, None
        # self.__C, self.__S = None, None
        self.__epoch = None

    def load(self, fileIn, date=None):
        """
        load GOCO06s fields
        :param fileIn: gif48 file and its path
        :param date: None, YYYY-MM, YYYY-MM-DD, YYYYMMDD, or [start, end].
        :return:
        """

        flag = 0
        self.__epoch = self.__parse_epoch(date)

        with open(fileIn) as f:
            content = f.readlines()
            pass

        flag = self.__header_GOCO06S(content)

        self.__read_GOCO06S(content[flag + 1:])

        return self

    def __parse_epoch(self, date):
        if date is None:
            return None

        if isinstance(date, (list, tuple)) and len(date) == 2:
            start = self.__parse_epoch(date[0])
            end = self.__parse_epoch(date[1])
            return start + (end - start) / 2

        if isinstance(date, datetime):
            return date

        if isinstance(date, str):
            text = date.strip()
            if len(text) == 7:
                start = datetime.strptime(text + "-01", "%Y-%m-%d")
                if start.month == 12:
                    end = datetime(start.year + 1, 1, 1)
                else:
                    end = datetime(start.year, start.month + 1, 1)
                return start + (end - start) / 2
            if len(text) == 8 and text.isdigit():
                return datetime.strptime(text, "%Y%m%d")
            return datetime.strptime(text, "%Y-%m-%d")

        raise TypeError("Unsupported GOCO06s epoch: %r" % (date,))

    @staticmethod
    def __years_between(epoch, t0):
        return (epoch - t0).total_seconds() / 86400.0 / 365.25

    def __header_GOCO06S(self, content: list):
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

    def __read_GOCO06S(self, content: list):

        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        maxdeg = int(self.maxDegree)

        stempC = np.zeros((maxdeg + 1, maxdeg + 1))
        stempS = np.zeros((maxdeg + 1, maxdeg + 1))
        stempsigmaC = np.zeros((maxdeg + 1, maxdeg + 1))
        stempsigmaS = np.zeros((maxdeg + 1, maxdeg + 1))
        current_L = None
        current_M = None
        current_tau = None

        for line in content:

            value = line.split()
            if len(value) == 0:
                continue

            key = value[0]

            # 只读取 gfc / gfct
            if key not in ("gfc", "gfct", "trnd", "acos", "asin"):
                continue

            if key in ("gfc", "gfct"):
                L = int(value[1])
                M = int(value[2])

                if L > maxdeg:
                    current_L = None
                    current_M = None
                    current_tau = None
                    continue

                stempC[L, M] = float(value[3].replace('D', 'E'))
                stempS[L, M] = float(value[4].replace('D', 'E'))
                stempsigmaC[L, M] = float(value[5].replace('D', 'E'))
                stempsigmaS[L, M] = float(value[6].replace('D', 'E'))

                current_L = L
                current_M = M
                current_tau = None
                if key == "gfct" and self.__epoch is not None and len(value) >= 8:
                    t0 = datetime.strptime(value[7], "%Y%m%d")
                    current_tau = self.__years_between(self.__epoch, t0)
                continue

            if self.__epoch is None or current_L is None or current_tau is None:
                continue

            c = float(value[3].replace('D', 'E'))
            s = float(value[4].replace('D', 'E'))
            if key == "trnd":
                factor = current_tau
            else:
                period = float(value[7].replace('D', 'E')) if len(value) >= 8 else 1.0
                phase = 2.0 * np.pi * current_tau / period
                factor = np.cos(phase) if key == "acos" else np.sin(phase)

            stempC[current_L, current_M] += c * factor
            stempS[current_L, current_M] += s * factor

        for i in range(maxdeg + 1):
            for j in range(i + 1):
                l.append(i)
                m.append(j)

                C.append(stempC[i, j])
                S.append(stempS[i, j])
                sigmaC.append(stempsigmaC[i, j])
                sigmaS.append(stempsigmaS[i, j])

        l = np.array(l, dtype=np.int64)
        m = np.array(m, dtype=np.int64)
        C = np.array(C, dtype=np.float64)
        S = np.array(S, dtype=np.float64)
        sigmaC = np.array(sigmaC, dtype=np.float64)
        sigmaS = np.array(sigmaS, dtype=np.float64)

        index = (l * (l + 1) // 2 + m)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[index] = C
        self._S[index] = S
        self._sigmaC[index] = sigmaC
        self._sigmaS[index] = sigmaS


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
