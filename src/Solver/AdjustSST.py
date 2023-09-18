
import pathlib
import h5py
from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType
from src.Auxilary.Outlier import Outlier
from src.Auxilary.KeplerOrbitElements import OrbitalElements
from tqdm import tqdm
import numpy as np
import scipy.linalg
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Preference.Pre_Interface import InterfaceConfig
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']         #指定默认字体（因为matplotlib默认为英文字体，汉字会使其乱码）
plt.rcParams['axes.unicode_minus']=False    #可显示‘-’负号

import os


class RangeRate:
    def __init__(self, arcNo: int, date_span):
        """init variable"""
        self.__rms_times = None
        self.__upper_RMS = None
        self.__kbr_fit = None
        self.__iterations = None
        self.__kbr_arc = None
        self.__arcNo = arcNo
        self.__date_span = date_span
        self._AdjustOrbitConfig = None
        self._RangeRateConfig = None
        self._PathfileConfig = None
        self._InterfaceConfig = None
        pass

    def configure(self, AdjustOrbitConfig:AdjustOrbitConfig, InterfaceConfig: InterfaceConfig,**kwargs):
        self._AdjustOrbitConfig = AdjustOrbitConfig
        self._InterfaceConfig = InterfaceConfig
        '''config adjust orbit'''
        self._RangeRateConfig = self._AdjustOrbitConfig.RangeRate()
        self._RangeRateConfig.__dict__.update(self._AdjustOrbitConfig.RangeRateConfig.copy())
        self._PathfileConfig = self._AdjustOrbitConfig.PathOfFiles()
        self._PathfileConfig.__dict__.update(self._AdjustOrbitConfig.PathOfFilesConfig.copy())

        kbr_arc = self._RangeRateConfig.ArcLength
        self.__rms_times = self._RangeRateConfig.OutlierTimes
        self.__upper_RMS = self._RangeRateConfig.OutlierLimit
        self.__kbr_fit = self._RangeRateConfig.ParameterFitting
        self.__iterations = self._RangeRateConfig.Iterations
        '''the arc length to Calibrate kbr: unit [hour]'''
        self.__kbr_arc = kbr_arc * 3600

        """load data"""
        filename = self._PathfileConfig.StateVectorDataTemp + '/' +\
                   self.__date_span[0] + '_' + self.__date_span[1] + '/' + str(self.__arcNo) + '_AB.hdf5'
        h5AB = h5py.File(filename, 'r')
        rd = ArcData(interfaceConfig=self._InterfaceConfig)
        KBR = rd.getData(arc=self.__arcNo, kind=EnumType.Payload.KBR)

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
        res_dir = self._PathfileConfig.RangeRateTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        res_filename = res_dir + '/RangeRate_' + str(self.__arcNo) + '.hdf5'
        self.__resh5 = h5py.File(res_filename, "w")
        return self

    def calibrate(self):
        obs = self.pre_residual()
        n = self.mean_motion_of_the_mid_point()

        '''divide arcs to Calibrate kbr individually'''
        node = [self.__Time[0]]
        t1 = self.__Time[0]
        while True:
            t2 = t1 + self.__kbr_arc
            if np.fabs(t2 - self.__Time[-1]) <= 0.5 * 3600:
                '''difference less than 0.5 hour, end'''
                node.append(t2)
                break
            elif len(self.__Time[self.__Time >= t2]) <= 100:
                '''the rest points are too short, e.g., less than 100: include the rest into the last arc'''
                node[-1] = self.__Time[-1] + 10  # +10, a little bigger than the last point to ensure all is included
            else:
                node.append(t2)
                t1 = t2

        '''attribute data into each kbr arc'''
        arclist_t, arclist_n, arclist_obs = [], [], []
        for i in range(len(node) - 1):
            index = (self.__Time < node[i + 1]) * (self.__Time >= node[i])
            arclist_t.append(self.__Time[index])
            arclist_n.append(n[index])
            arclist_obs.append(obs[index])

        '''Calibrate kbr arc by arc'''
        post_t, post_residual = np.zeros(0), np.zeros(0)
        dm_list = []
        for i in tqdm(range(len(arclist_t)), desc='SST rang-rate calibration: '):
            t, n, obs = arclist_t[i], arclist_n[i], arclist_obs[i]
            kf = FitSST(t, n, option=self.__kbr_fit)
            dm = kf.get_design_matrix()

            for iter in range(self.__iterations):
                residual = kf.get_residual(dm.copy(), obs)
                index = Outlier(rms_times=self.__rms_times, upper_RMS=self.__upper_RMS).remove(t, residual)

                if any(index) == False:
                    t = t[index]
                    dm = dm[index, :]
                    obs = residual[index]
                    break

            post_t = np.append(post_t, t)
            post_residual = np.append(post_residual, obs)
            dm_list.append(dm)

        design_matrix = scipy.linalg.block_diag(*dm_list)

        '''make a record'''
        h5 = self.__resh5
        h5.create_dataset('post_t', data=post_t)
        h5.create_dataset('post_residual', data=post_residual)
        h5.create_dataset('pre_t', data=self.__Time)
        h5.create_dataset('pre_residual', data=self.pre_residual())
        h5.create_dataset('design_matrix', data=design_matrix)
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


class FitSST:

    def __init__(self, t: np.ndarray, n: np.ndarray, option):
        """
        :param t: epoch of the kbr data  [second]
        :param n: mean motion of the sat. [kepler orbital element]
        :param option: to choose which method is applied.
        """
        method = {
            EnumType.SSTFitOption.biasC0C1.name: self.__func1,
            EnumType.SSTFitOption.biasC0C1C2.name: self.__func2,
            EnumType.SSTFitOption.biasC0C1C2C3.name: self.__func3,
            EnumType.SSTFitOption.biasC0_OneCPR.name: self.__func4,
            EnumType.SSTFitOption.biasC0C1_OneCPR.name: self.__func5,
            EnumType.SSTFitOption.biasC0C1C2_OneCPR.name: self.__func6,
            EnumType.SSTFitOption.twice_biasC0C1_OneCPR.name: self.__func7,
            EnumType.SSTFitOption.twice_biasC0C1C2_OneCPR.name: self.__func8,
            EnumType.SSTFitOption.test.name: self.__functest
        }

        self.__DesignMatrix = method[option](t, n)
        pass

    def __func1(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        leng = np.shape(t)[0]

        term1 = np.ones(leng, dtype=np.float64)
        term2 = t

        DesignMatrix = np.array([term1, term2]).transpose()

        return DesignMatrix

    def __func2(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt+ct^2
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        leng = np.shape(t)[0]

        term1 = np.ones(leng, dtype=np.float64)
        term2 = t
        term3 = t ** 2

        DesignMatrix = np.array([term1, term2, term3]).transpose()

        return DesignMatrix

    def __func3(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt+ct^2+dt^3
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        leng = np.shape(t)[0]

        term1 = np.ones(leng, dtype=np.float64)
        term2 = t
        term3 = t ** 2
        term4 = t ** 3

        DesignMatrix = np.array([term1, term2, term3, term4]).transpose()

        return DesignMatrix

    def __func4(self, t: np.ndarray, n: np.ndarray):
        """
        a+(E+Ft)cosnt+(G+Ht)sinnt
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        nt = n * t
        leng = np.shape(t)[0]

        term1 = np.ones(leng, dtype=np.float64)
        term2 = np.cos(nt)
        term3 = t * np.cos(nt)
        term4 = np.sin(nt)
        term5 = t * np.sin(nt)

        DesignMatrix = np.array([term1, term2, term3, term4, term5]).transpose()

        return DesignMatrix

    def __func5(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt + (E+Ft)cosnt+(G+Ht)sinnt
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        nt = n * t
        leng = np.shape(t)[0]

        term1 = np.ones(leng, dtype=np.float64)
        term2 = t

        term3 = np.cos(nt)
        term4 = t * np.cos(nt)
        term5 = np.sin(nt)
        term6 = t * np.sin(nt)

        DesignMatrix = np.array([term1, term2, term3, term4, term5, term6]).transpose()

        return DesignMatrix

    def __func6(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt + ct^2 + (E+Ft)cosnt+(G+Ht)sinnt
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        nt = n * t
        leng = np.shape(t)[0]

        term1 = np.ones(leng, dtype=np.float64)
        term2 = t
        term3 = t ** 2

        term4 = np.cos(nt)
        term5 = t * np.cos(nt)
        term6 = np.sin(nt)
        term7 = t * np.sin(nt)

        DesignMatrix = np.array([term1, term2, term3, term4, term5, term6, term7]).transpose()

        return DesignMatrix

    def __functest(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt + ct^2 + dt^3 + (E+Ft+Qt^2)cosnt+(G+Ht+F^2)sinnt
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        nt = n * t
        leng = np.shape(t)[0]

        term1 = np.ones(leng, dtype=np.float64)
        term2 = t
        term3 = t ** 2
        term4 = t * t * t

        term5 = np.cos(nt)
        term6 = t * np.cos(nt)
        term7 = t * t * np.cos(nt)
        term8 = np.sin(nt)
        term9 = t * np.sin(nt)
        term10 = t * t * np.sin(nt)

        DesignMatrix = np.array([term1, term2, term3, term4, term5, term6, term7, term8, term9, term10]).transpose()

        return DesignMatrix

    def __func7(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt + (E+Ft)cosnt+(G+Ht)sinnt
        !! the bias will be estimated twice per time
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        nt = n * t
        leng = np.shape(t)[0]

        half_leng = int(leng / 2)

        term1 = np.ones(half_leng, dtype=np.float64)
        term2 = t[0:half_leng]
        term1 = np.append(term1, np.zeros(leng - half_leng))
        term2 = np.append(term2, np.zeros(leng - half_leng))

        term3 = np.ones(leng - half_leng, dtype=np.float64)
        term4 = t[half_leng:]
        term3 = np.append(np.zeros(half_leng), term3)
        term4 = np.append(np.zeros(half_leng), term4)

        term5 = np.cos(nt)
        term6 = t * np.cos(nt)
        term7 = np.sin(nt)
        term8 = t * np.sin(nt)

        DesignMatrix = np.array([term1, term2, term3, term4, term5, term6, term7, term8]).transpose()

        return DesignMatrix

    def __func8(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt + ct^2 + (E+Ft)cosnt+(G+Ht)sinnt
        !! the bias will be estimated twice per time
        :param t: one-dimension
        :param n: one-dimension
        :return:
        """
        t = t - t[0]
        nt = n * t
        leng = np.shape(t)[0]

        half_leng = int(leng / 2)

        term1 = np.ones(half_leng, dtype=np.float64)
        term2 = t[0:half_leng]
        term3 = t[0:half_leng] ** 2
        term1 = np.append(term1, np.zeros(leng - half_leng))
        term2 = np.append(term2, np.zeros(leng - half_leng))
        term3 = np.append(term3, np.zeros(leng - half_leng))

        term4 = np.ones(leng - half_leng, dtype=np.float64)
        term5 = t[half_leng:]
        term6 = t[half_leng:] ** 2
        term4 = np.append(np.zeros(half_leng), term4)
        term5 = np.append(np.zeros(half_leng), term5)
        term6 = np.append(np.zeros(half_leng), term6)

        term7 = np.cos(nt)
        term8 = t * np.cos(nt)
        term9 = np.sin(nt)
        term10 = t * np.sin(nt)

        DesignMatrix = np.array([term1, term2, term3, term4, term5, term6, term7, term8, term9, term10]).transpose()

        return DesignMatrix

    def get_design_matrix(self):
        return self.__DesignMatrix

    @staticmethod
    def get_residual(DesignMatrix: np.ndarray, obs: np.ndarray):
        """
        solve the least-square problem and get the residual of the observations
        :param DesignMatrix: ...
        :param obs: observations to be fitted
        :return: residual
        """

        x = np.linalg.lstsq(DesignMatrix, obs, rcond=None)

        residual = obs - DesignMatrix @ x[0]

        return residual