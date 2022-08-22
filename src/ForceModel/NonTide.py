from src.ForceModel.BaseForceCS import BaseForceCS
from src.ForceModel.LoadSH import LoadSH
from src.Preference.EnumType import AODtype
from src.Preference.Pre_ForceModel import PathOfFiles, NonTideConfig
import src.Preference.Pre_Constants as Pre_Constants
from src.Frame.Frame import Frame
import os
import datetime
import numpy as np


class AbsNonTide(BaseForceCS):

    def __init__(self):
        super(AbsNonTide, self).__init__()

    def setPar(self, MaxDeg: int, TS: int, FileDir: str, AOD_type: AODtype):
        pass

    def getCS(self):
        pass


class NonTide(AbsNonTide):

    def __init__(self):
        super(NonTide, self).__init__()
        self.__gpstime = -1000
        self.__ld = None
        self.__TS = 3
        self.__MaxDeg = 100
        self.__previousAOD, self.__nextAOD = None, None

    def setPar(self, MaxDeg: int = NonTideConfig.MaxDeg, TS: int = NonTideConfig.TS,
               FileDir = PathOfFiles.AOD, AOD_type: AODtype = NonTideConfig.kind):
        """
        basic configuration
        :param AOD_type: A or O or A+O
        :param MaxDeg: Max degree of spherical harmonic to be extracted
        :param TS: temporal sampling rate of the AOD product, for instance, TS=1, TS=3, TS=6, unit[hour].
        :param FileDir: Directory of the AOD product.
        :return:
        """
        self.__MaxDeg = MaxDeg
        self.__TS = TS
        self.__ld = LoadAOD().load(FileDir).setType(kind=AOD_type)

        return self

    def getCS(self):
        """
        get CS at given time
        :param gpsTime: gps time in terms of sec since J2000
        :return:
        """
        gpsTime = self._time.getGPS_from_epoch2000
        mjd = gpsTime / 86400 + Pre_Constants.J2000_MJD
        # mjd = gpsTime
        cal = Frame.mjd2cal(mjd)

        day = datetime.datetime(year=cal[0], month=cal[1], day=cal[2], hour=int(cal[3]))

        if gpsTime<self.__gpstime or gpsTime>self.__gpstime+self.__TS*3600:
            hr_previous = int(cal[3]) % self.__TS
            hr_next = self.__TS - hr_previous
            previous_date = day + datetime.timedelta(hours=-hr_previous)
            next_date = day + datetime.timedelta(hours=hr_next)

            self.__gpstime = (Frame.cal2mjd(IY=previous_date.year, IM=previous_date.month, ID=previous_date.day,
                                           IH=previous_date.hour, MIN=0, SEC=0) - Pre_Constants.J2000_MJD)*86400

            rd = self.__ld.setTime(date=previous_date.date().__str__(), epoch=previous_date.time().__str__())
            self.__previousAOD = np.array(rd.getCS(self.__MaxDeg))

            rd = self.__ld.setTime(date=next_date.date().__str__(), epoch=next_date.time().__str__())
            self.__nextAOD = np.array(rd.getCS(self.__MaxDeg))

        '''linear interpolation'''
        res = (gpsTime - self.__gpstime)/(self.__TS*3600)*(self.__nextAOD - self.__previousAOD) + self.__previousAOD

        return res[0], res[1]


class LoadAOD(LoadSH):
    """
    This class is used to read stokes coefficients at given epoch from AOD product.
    """

    def __init__(self):
        LoadSH.__init__(self)
        self.__kind = None
        self.__AODdir = None
        self.__box = {AODtype.ATM: 'atm',
                      AODtype.OCN: 'ocn',
                      AODtype.OBA: 'oba',
                      AODtype.GLO: 'glo'}
        self.__epoch = None

    def load(self, fileIn = PathOfFiles.AOD):
        """
        set the directory of AOD products to be read
        :param fileIn:
        :return:
        """
        self.__AODdir = fileIn
        return self

    def setTime(self, date: str = NonTideConfig.date, epoch: str = NonTideConfig.epoch):
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

    def setType(self, kind: AODtype = NonTideConfig.kind):
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

