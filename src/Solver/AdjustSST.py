import pathlib
import h5py
import src.Preference.Pre_AdjustOrbit as Pre_AdjustOrbit
from src.Interface.ArcOperation import ArcData
import src.Preference.Pre_Interface as Pre_Interface
import src.Preference.EnumType as EnumType
from src.Auxilary.KeplerOrbitElements import OrbitalElements
import numpy as np


class RangeRate:
    def __init__(self, arcNo: int, date_span=Pre_Interface.date_span):
        """init variable"""
        self.__rms_times = None
        self.__upper_RMS = None
        self.__kbr_fit = None
        self.__iterations = None
        self.__kbr_arc = None
        """load data"""
        filename = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp).\
            joinpath(date_span[0] + '_' + date_span[1]).joinpath(str(arcNo) + '_AB.hdf5')
        h5AB = h5py.File(filename, 'r')
        rd = ArcData(date_span=date_span)
        KBR = rd.getData(arc=arcNo, kind=EnumType.Payload.KBR)

        kbr_time = KBR[:, 0]
        self.__kbr_r = KBR[:, 2]
        state_time = list(h5AB['t'][:])

        '''find the common time of orbit and kbr data'''
        new_index = [state_time.index(x) for x in kbr_time]
        self.__Time = h5AB['t'][:][new_index]
        self.__PosA = h5AB['r'][:][new_index, 0:3, 0]
        self.__PosB = h5AB['r'][:][new_index, 3:6, 0]
        self.__VelA = h5AB['v'][:][new_index, 0:3, 0]
        self.__VelB = h5AB['v'][:][new_index, 3:6, 0]

        '''define the directory of range-rate output'''
        res_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.RangeRateTemp). \
            joinpath(date_span[0] + '_' + date_span[1])
        if not res_dir.is_dir():
            res_dir.mkdir(exist_ok=True)
        res_filename = res_dir.joinpath('RangeRate_' + str(arcNo) + '.hdf5')
        self.__resh5 = h5py.File(res_filename, "w")

        pass

    def configure(self, **kwargs):
        kbr_arc = kwargs.get('kbr_arc', Pre_AdjustOrbit.RangeRate.ArcLength)
        self.__rms_times = kwargs.get('rms_times', Pre_AdjustOrbit.RangeRate.OutlierTimes)
        self.__upper_RMS = kwargs.get('upper_RMS', Pre_AdjustOrbit.RangeRate.OutlierLimit)
        self.__kbr_fit = kwargs.get('kbr_fit', Pre_AdjustOrbit.RangeRate.ParameterFitting)
        self.__iterations = kwargs.get('iterations', Pre_AdjustOrbit.RangeRate.Iterations)
        self.__kbr_arc = kbr_arc * 3600
        return self

    def calibrate(self):
        pass

    def pre_residual(self):
        diff_pos = self.__PosA - self.__PosB
        diff_vel = self.__VelA - self.__VelB

        diff_pos_norm = np.sqrt(diff_pos[:, 0] ** 2 + diff_pos[:, 1] ** 2 + diff_pos[:, 2] ** 2)
        e12 = diff_pos / diff_pos_norm[:, None]

        sst_nominal = diff_vel[:, 0] * e12[:, 0] + diff_vel[:, 1] * e12[:, 1] + diff_vel[:, 2] * e12[:, 2]
        return self.__kbr_r - sst_nominal

    def mean_motion_of_the_mid_point(self):
        mid_pos = (self.__PosA + self.__PosB) / 2
        mid_vel = (self.__VelA + self.__VelB) / 2

        oe = OrbitalElements()
        oe.setState(x=mid_pos[:, 0], y=mid_pos[:, 1], z=mid_pos[:, 2], vx=mid_vel[:, 0], vy=mid_vel[:, 1],
                    vz=mid_vel[:, 2])

        return oe.getMeanMotion()


