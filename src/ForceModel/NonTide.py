from src.ForceModel.BaseGravCS import BaseGravCS
import src.Preference.Pre_Constants as Pre_Constants
from src.Interface.LoadSH import LoadNonTide, LoadGLO
from src.Frame.Frame import Frame
import datetime
import numpy as np
from src.Preference.Pre_ForceModel import ForceModelConfig


class AbsNonTide(BaseGravCS):

    def __init__(self):
        super(AbsNonTide, self).__init__()

    def configure(self, FMConfig: ForceModelConfig):
        pass

    def getCS(self):
        pass


class AODRL06(AbsNonTide):

    def __init__(self):
        super(AODRL06, self).__init__()
        self.__gpstime = -1000
        self.__ld = None
        self.__TS = 3
        self.__MaxDeg = 100
        self.__FMConfig = None
        self._NonTideConfig = None
        self._pathConfig = None
        self.__previousAOD, self.__nextAOD = None, None

    def configure(self, FMConfig: ForceModelConfig):
        """
        basic configuration
        :param AOD_type: A or O or A+O
        :param MaxDeg: Max degree of spherical harmonic to be extracted
        :param TS: temporal sampling rate of the AOD product, for instance, TS=1, TS=3, TS=6, unit[hour].
        :param FileDir: Directory of the AOD product.
        :return:
        """
        self.__FMConfig = FMConfig
        '''config AOD'''
        self._NonTideConfig = self.__FMConfig.NonTide()
        self._NonTideConfig.__dict__.update(self.__FMConfig.NonTideConfig.copy())
        '''config path'''
        self._pathConfig = self.__FMConfig.PathOfFiles()
        self._pathConfig.__dict__.update(self.__FMConfig.PathOfFilesConfig.copy())

        FileDir = self._pathConfig.AOD
        AOD_type = self._NonTideConfig.kind
        self.__MaxDeg = self._NonTideConfig.MaxDeg
        self.__TS = self._NonTideConfig.TS
        self.__ld = LoadGLO().load(FileDir).setType(kind=AOD_type)

        return self

    def getCS(self):
        """
        get CS at given time
        :param gpsTime: gps time in terms of sec since J2000
        :return:
        """
        gpsTime = float(self._time.getGPS_from_epoch2000)
        mjd = gpsTime / 86400 + Pre_Constants.J2000_MJD
        # mjd = gpsTime
        cal = Frame.mjd2cal(mjd)

        day = datetime.datetime(year=cal[0], month=cal[1], day=cal[2], hour=int(cal[3]))

        if gpsTime < self.__gpstime or gpsTime > self.__gpstime + self.__TS * 3600:
            hr_previous = int(cal[3]) % self.__TS
            hr_next = self.__TS - hr_previous
            previous_date = day + datetime.timedelta(hours=-hr_previous)
            next_date = day + datetime.timedelta(hours=hr_next)

            self.__gpstime = (Frame.cal2mjd(IY=previous_date.year, IM=previous_date.month, ID=previous_date.day,
                                            IH=previous_date.hour, MIN=0, SEC=0) - Pre_Constants.J2000_MJD) * 86400

            rd = self.__ld.setTime(date=previous_date.date().__str__(), epoch=previous_date.time().__str__())
            self.__previousAOD = np.array(rd.getCS(self.__MaxDeg))

            rd = self.__ld.setTime(date=next_date.date().__str__(), epoch=next_date.time().__str__())
            self.__nextAOD = np.array(rd.getCS(self.__MaxDeg))

        '''linear interpolation'''
        res = (gpsTime - self.__gpstime) / (self.__TS * 3600) * \
              (self.__nextAOD - self.__previousAOD) + self.__previousAOD

        return res[0], res[1]

