from src.ForceModel.BaseSecondDerivative import BaseSecondDerivative
from src.Preference.Pre_ForceModel import NonConservative, PathOfFiles
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Frame.Frame import Frame
import src.Preference.EnumType as EnumType
import h5py
import pathlib
import numpy as np


class AbsNonConservative(BaseSecondDerivative):

    def __init__(self,  sat: EnumType.SatID, range=NonConservative.date_span, arcNo=NonConservative.arcNo):
        super(AbsNonConservative, self).__init__()
        self._GPStime = None


class NonConByAcc(AbsNonConservative):

    def __init__(self, sat: EnumType.SatID, range=NonConservative.date_span, arcNo=NonConservative.arcNo):
        """
        Load the saved-already acc and dadp, and output the acc/dadp for the given time epoch.
        :param fileDir: acc and dadp have been computed beforehand and saved in given directory.
        :param range: a two-element tuple recording the range of the Acc timeseries, in GPS second since J2000.
                        for example, (254546668,254567889)
        """
        super().__init__(range=range, sat=sat, arcNo=arcNo)
        self.__dataTemp = None
        self.__GPStime = None
        self.sat = sat
        self._range = range
        self._arcNo = arcNo

    def configure(self, **kwargs):
        self.__dataTemp = kwargs.get('dataTemp', PathOfFiles.temp_non_conservative_data)
        self.__loadAccelerometer()
        return self

    def __loadAccelerometer(self):
        filename = pathlib.Path(self.__dataTemp).joinpath(
            self._range[0] + '_' + self._range[1] + '_' + self.sat.name + '_' + str(self._arcNo) + '.hdf5')
        h5data = h5py.File(filename, "r")

        self.__time = h5data['time'][()]
        self.__acc = h5data['acc'][()]
        self.__dadp = h5data['dadp'][()]
        h5data.close()

    def setTime(self, time: Frame):
        """
        :param time: GPS seconds since J2000
        :return:
        """
        '''assuring the time is located within the range'''
        err = 1e-3 # GPS time is not that accurate because of the increment of the time
        assert self.__time[0] - err <= time.getGPS_from_epoch2000 <= self.__time[-1] + err, 'ACC instrument does not cover the given time '
        # TODO: efficiency
        self.__idx = np.abs(time.getGPS_from_epoch2000 - self.__time).argmin()
        return self

    def getAcceleration(self):
        """
        :return: acc [m/s][GCRS]
        """
        return self.__acc[self.__idx, :]

    def getDaDp(self):
        """

        :return: dadp [GCRS]
        """
        return self.__dadp[self.__idx, :, :]

