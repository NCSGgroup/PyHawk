
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
from src.ODE.Instance.GravimetryODE import GravimetryODE
from src.Preference.Pre_ODE import ODEConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Frame.Frame import Frame, EOP
from src.Preference.Pre_Frame import FrameConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False    #可显示‘-’负号

import os


class RangeRate:
    _EMPIRICAL_STAGES = {
        'pre': ('empirical_prefit_t', 'empirical_prefit_residual'),
        'post': ('empirical_postfit_t', 'empirical_postfit_residual'),
    }

    def __init__(self, arcNo: int, date_span):
        """init variable"""
        self.__rms_times = None
        self.__upper_RMS = None
        self.__upper_obs = None
        self.__kbr_fit = None
        self.__onecpr_outlier_only = True
        self.__iterations = None
        self.__kbr_arc = None
        self.__arcNo = arcNo
        self.__date_span = date_span
        self._AdjustOrbitConfig = None
        self._RangeRateConfig = None
        self._PathfileConfig = None
        self._InterfaceConfig = None
        pass

    def configure(self, SerDer, ODEConfig:ODEConfig, ParameterConfig: ParameterConfig,
                  AdjustOrbitConfig:AdjustOrbitConfig, FrameConfig:FrameConfig,
                  InterfaceConfig: InterfaceConfig,FMConfig: ForceModelConfig, **kwargs):
        self._AdjustOrbitConfig = AdjustOrbitConfig
        self._InterfaceConfig = InterfaceConfig
        eop = EOP().configure(frameConfig=FrameConfig).load()
        self._fr = Frame(eop).configure(frameConfig=FrameConfig)
        '''config Force model'''
        self.FMConfig = FMConfig
        '''config adjust orbit'''
        self._RangeRateConfig = self._AdjustOrbitConfig.RangeRate()
        self._RangeRateConfig.__dict__.update(self._AdjustOrbitConfig.RangeRateConfig.copy())
        self._PathfileConfig = self._AdjustOrbitConfig.PathOfFiles()
        self._PathfileConfig.__dict__.update(self._AdjustOrbitConfig.PathOfFilesConfig.copy())
        """load data"""
        filename = self._PathfileConfig.StateVectorDataTemp + '/' +\
                   self.__date_span[0] + '_' + self.__date_span[1] + '/' + str(self.__arcNo) + '_AB.hdf5'
        h5AB = h5py.File(filename, 'r')

        '''config ODE'''
        self.ODEConfig = ODEConfig
        '''config TransitionMatrix'''
        self.ParameterConfig = ParameterConfig
        """step 3: instance second Derivative"""
        rd = ArcData(interfaceConfig=self._InterfaceConfig)
        self._rd = rd

        self._secDer = SerDer(self._fr, rd).configures\
            (arc=self.__arcNo, FMConfig=self.FMConfig,
             ParConfig=self.ParameterConfig, FrameConfig=FrameConfig).setInitData(kind=EnumType.Payload.GNV)
        times = self._secDer.getTimes()

        self._startTime, self._startPos, self._startVel = h5AB['t'][:][4], \
                        np.vstack((h5AB['r'][:][4, 0:3, 0], h5AB['r'][:][4, 3:6, 0])), \
                        np.vstack((h5AB['v'][:][4, 0:3, 0], h5AB['v'][:][4, 3:6, 0]))
        self._ode = GravimetryODE().configure(ODEConfig=self.ODEConfig, ParameterConfig=self.ParameterConfig).\
            setDataTime(times).setParNum().setInitial(self._startTime, self._startPos.copy(), self._startVel.copy()).\
            set2ndDerivative(self._secDer)

        self._ode_res = self._ode.propagate()
        t,r,v = self._ode_res[0], self._ode_res[1], self._ode_res[2]

        kbr_arc = self._RangeRateConfig.ArcLength
        self.__rms_times = self._RangeRateConfig.OutlierTimes
        self.__upper_RMS = self._RangeRateConfig.OutlierRMS
        self.__upper_obs = self._RangeRateConfig.OutlierObs
        self.__kbr_fit = self._RangeRateConfig.ParameterFitting
        self.__onecpr_outlier_only = bool(
            getattr(self._RangeRateConfig, 'OneCPROutlierOnly', True)
        )
        self.__iterations = self._RangeRateConfig.Iterations
        '''the arc length to Calibrate kbr: unit [hour]'''
        self.__kbr_arc = kbr_arc * 3600

        rd = ArcData(interfaceConfig=self._InterfaceConfig)
        KBR = rd.getData(arc=self.__arcNo, kind=EnumType.Payload.KBR)

        kbr_time = KBR[:, 0]
        self.__kbr_r = KBR[:, 2]

        # state_time = h5AB['t'][:]
        state_time = t

        commonTime = set(KBR[:, 0].astype(np.int64))
        commonTime = commonTime & set(state_time.astype(np.int64))
        commonTime = np.array(list(commonTime))
        commonTime.sort()

        '''find the common time of orbit and kbr data'''
        orbit_index = [list(state_time).index(x) for x in commonTime]
        kbr_index = [list(kbr_time).index(x) for x in commonTime]
        self.Time = t[orbit_index]
        self.__PosA = r[orbit_index, 0:3, 0]
        self.__PosB = r[orbit_index, 3:6, 0]
        self.__VelA = v[orbit_index, 0:3, 0]
        self.__VelB = v[orbit_index, 3:6, 0]
        self.__kbr_r = KBR[kbr_index, 2]
        '''define the directory of range-rate output'''
        res_dir = self._PathfileConfig.RangeRateTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        res_filename = res_dir + '/RangeRate_' + str(self.__arcNo) + '.hdf5'
        self.__resh5 = h5py.File(res_filename, "w")
        return self

    @staticmethod
    def _build_arc_nodes(time, arc_length, close_tolerance=1800,
                         min_tail_points=100, end_margin=10):
        """Build fitting intervals that cover every KBRR epoch.

        A short final remainder is merged into the current interval.  In
        particular, an input arc shorter than ``arc_length`` still produces
        one fitting interval instead of an empty node list.
        """
        time = np.asarray(time)
        if time.ndim != 1 or time.size == 0:
            raise ValueError('KBRR time must be a non-empty one-dimensional array')
        if not np.all(np.isfinite(time)):
            raise ValueError('KBRR time contains non-finite values')
        if np.any(np.diff(time) < 0):
            raise ValueError('KBRR time must be sorted in ascending order')
        if arc_length <= 0:
            raise ValueError('KBRR fitting arc length must be positive')

        nodes = [time[0]]
        end_time = time[-1]
        while True:
            next_node = nodes[-1] + arc_length
            if next_node >= end_time:
                nodes.append(end_time + end_margin)
                break

            tail_points = np.count_nonzero(time >= next_node)
            if (end_time - next_node <= close_tolerance or
                    tail_points <= min_tail_points):
                nodes.append(end_time + end_margin)
                break

            nodes.append(next_node)

        return nodes

    def calibrate(self):
        """Use an arc-wise empirical model to screen KBRR O-C outliers.

        With ``OneCPROutlierOnly`` enabled, the trend/1CPR fit is used only
        to expose abnormal epochs.  The retained, otherwise unmodified O-C
        is written to the downstream interface consumed by LGD and NEQ.
        """
        obs = self.pre_residual()

        n = self.mean_motion_of_the_mid_point()

        '''divide arcs to Calibrate kbr individually'''
        node = self._build_arc_nodes(self.Time, self.__kbr_arc)

        '''attribute data into each kbr arc'''
        arclist_t, arclist_n, arclist_obs = [], [], []
        for i in range(len(node) - 1):
            index = ((self.Time < node[i + 1]) &
                     (self.Time >= node[i]))
            arclist_t.append(self.Time[index])
            arclist_n.append(n[index])
            arclist_obs.append(obs[index])

        '''Calibrate kbr arc by arc'''
        pre_t, pre_residual = np.zeros(0), np.zeros(0)
        post_t, post_residual = np.zeros(0), np.zeros(0)
        detection_t, detection_residual = np.zeros(0), np.zeros(0)
        dm_list = []
        for i in tqdm(range(len(arclist_t)), desc='SST rang-rate calibration: '):
            t = arclist_t[i]
            n = arclist_n[i]
            source_obs = arclist_obs[i]
            if len(t) == 0:
                continue
            kf = FitSST(t, n, option=self.__kbr_fit)
            dm = kf.get_design_matrix()

            for iter in range(self.__iterations):
                residual = kf.get_residual(dm.copy(), source_obs)
                index = Outlier(rms_times=self.__rms_times, upper_RMS=self.__upper_RMS,
                                upper_obs=self.__upper_obs).remove(t, residual)
                if np.all(index):
                    break
                t = t[index]
                dm = dm[index, :]
                source_obs = source_obs[index]
                if not np.any(index) or len(t) == 0:
                    break

            if len(t) == 0:
                continue
            fit_residual = kf.get_residual(dm.copy(), source_obs)
            downstream_residual = (
                source_obs if self.__onecpr_outlier_only else fit_residual
            )
            pre_t = np.append(pre_t, t)
            pre_residual = np.append(pre_residual, source_obs)
            post_t = np.append(post_t, t)
            post_residual = np.append(post_residual, downstream_residual)
            detection_t = np.append(detection_t, t)
            detection_residual = np.append(
                detection_residual, fit_residual
            )
            dm_list.append(dm)

        design_matrix = scipy.linalg.block_diag(*dm_list)

        '''make a record'''
        h5 = self.__resh5
        h5.create_dataset('post_t', data=post_t)
        # h5.create_dataset('post_residual', data=pre_residual)
        h5.create_dataset('post_residual', data=post_residual)
        h5.create_dataset('pre_t', data=self.Time)
        h5.create_dataset('pre_residual', data=self.pre_residual())
        h5.create_dataset('design_matrix', data=design_matrix)
        h5.create_dataset('empirical_prefit_t', data=post_t)
        h5.create_dataset('empirical_prefit_residual', data=pre_residual)
        h5.create_dataset('empirical_postfit_t', data=post_t)
        h5.create_dataset('empirical_postfit_residual', data=post_residual)
        h5.create_dataset('retained_oc_t', data=post_t)
        h5.create_dataset('retained_oc_residual', data=pre_residual)
        h5.create_dataset('onecpr_detection_t', data=detection_t)
        h5.create_dataset(
            'onecpr_detection_residual', data=detection_residual
        )
        h5.attrs['onecpr_role'] = (
            'outlier_detection_only' if self.__onecpr_outlier_only
            else 'subtract_from_observation'
        )
        h5.attrs['downstream_observation'] = (
            'retained_o_c_without_trend_or_onecpr_subtraction'
            if self.__onecpr_outlier_only
            else 'empirical_postfit_residual'
        )
        h5.attrs['residual_stage_semantics'] = (
            'empirical pre-fit: retained raw O-C; '
            'empirical post-fit: downstream observation selected by onecpr_role; '
            'onecpr detection residual: fit residual used only for screening; '
            'not the spherical-harmonic gravity-field post-fit'
        )
        h5.close()

        pass

    @staticmethod
    def _load_qc_epoch_mask(h5, size):
        """Load the optional non-destructive epoch-QC selection mask."""
        if 'arc_qc_epoch_keep' not in h5:
            return np.ones(size, dtype=bool)
        keep = np.asarray(h5['arc_qc_epoch_keep'][()], dtype=bool).reshape(-1)
        if keep.size != size:
            raise ValueError(
                'arc_qc_epoch_keep length mismatch: %d vs %d'
                % (keep.size, size)
            )
        return keep

    @classmethod
    def load_empirical_residual(cls, filename, stage='Pre', apply_qc=True):
        """Load an aligned AdjustSST residual series and optional epoch QC."""
        normalized = str(stage).strip().lower().replace('-', '')
        if normalized == 'prefit':
            normalized = 'pre'
        elif normalized == 'postfit':
            normalized = 'post'
        if normalized not in cls._EMPIRICAL_STAGES:
            raise ValueError(
                "AdjustSST stage should be 'Pre' or 'Post', got %r" % stage
            )

        with h5py.File(filename, 'r') as h5:
            role = h5.attrs.get('onecpr_role', '')
            if isinstance(role, bytes):
                role = role.decode('utf-8')
            if normalized == 'post' and role == 'outlier_detection_only':
                time_name, residual_name = (
                    'retained_oc_t', 'retained_oc_residual'
                )
            else:
                time_name, residual_name = cls._EMPIRICAL_STAGES[normalized]
            missing = [
                name for name in (time_name, residual_name) if name not in h5
            ]
            if missing:
                raise KeyError(
                    '%s does not contain the AdjustSST %s interface: %s'
                    % (filename, normalized, ', '.join(missing))
                )
            time = np.asarray(h5[time_name][()], dtype=np.float64)
            residual = np.asarray(h5[residual_name][()], dtype=np.float64)
            keep = (
                cls._load_qc_epoch_mask(h5, time.size)
                if apply_qc else np.ones(time.size, dtype=bool)
            )

        if time.shape != residual.shape:
            raise ValueError(
                'AdjustSST %s time/residual shape mismatch: %s vs %s'
                % (normalized, time.shape, residual.shape)
            )
        return time[keep], residual[keep]

    @classmethod
    def load_empirical_design_matrix(cls, filename, apply_qc=True):
        """Load empirical nuisance columns aligned with downstream epochs.

        A detection-only OneCPR fit contributes no nuisance parameters to
        the normal equation; its stored matrix remains available for QC.
        """
        with h5py.File(filename, 'r') as h5:
            if 'design_matrix' not in h5:
                raise KeyError('%s does not contain design_matrix' % filename)
            matrix_shape = h5['design_matrix'].shape
            if len(matrix_shape) != 2:
                raise ValueError('design_matrix must be two-dimensional')
            role = h5.attrs.get('onecpr_role', '')
            if isinstance(role, bytes):
                role = role.decode('utf-8')
            if role == 'outlier_detection_only':
                matrix = np.empty((matrix_shape[0], 0), dtype=np.float64)
            else:
                matrix = np.asarray(
                    h5['design_matrix'][()], dtype=np.float64
                )
            keep = (
                cls._load_qc_epoch_mask(h5, matrix.shape[0])
                if apply_qc else np.ones(matrix.shape[0], dtype=bool)
            )
        return matrix[keep, :]

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

    def orbit_calibration(self):
        self._ode = GravimetryODE().configure(ODEConfig=self.ODEConfig,
                                              ParameterConfig=self.ParameterConfig).setParNum(). \
            setInitial(self._startTime, self._startPos.copy(), self._startVel.copy()). \
            set2ndDerivative(self._secDer)
        pass


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
            EnumType.SSTFitOption.test.name: self.__functest,
            EnumType.SSTFitOption.biasC0C1C2C3_OneCPR.name: self.__func10,
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
        a+bt + ct^2 + (E+Ft+Qt^2)cosnt+(G+Ht+F^2)sinnt
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

    def __func10(self, t: np.ndarray, n: np.ndarray):
        """
        a+bt + ct^2 + (E+Ft+Qt^2)cosnt+(G+Ht+F^2)sinnt
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
