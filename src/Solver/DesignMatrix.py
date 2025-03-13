from src.Preference.EnumType import SSTObserve
from src.Solver.AdjustDM import AdjustDM
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.Solver.Parameterization import Parameterization
from src.Auxilary.SSTpartial import SSTpartial
from src.Preference.Pre_Solver import SolverConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_ODE import ODEConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_Frame import FrameConfig
from src.Auxilary.KeplerOrbitElements import OrbitalElements
from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType
from src.Solver.AdjustSST import FitSST
from tqdm import tqdm
import numpy as np
import scipy.linalg
import pathlib
import h5py
import os


class GravityDesignMat:
    """
    Gravity inversion in terms of spherical harmonic.
    """
    def __init__(self, arcNo: int, sat, date_span):
        self._arcNo = arcNo
        self.__data_span = date_span
        self._satName = None
        self.__sat = [key for key, value in sat.items() if value]
        for sat in self.__sat:
            if self._satName is None:
                self._satName = sat
            else:
                self._satName = self._satName + sat
        self._res = None
        self._accConfig = None
        self._solverConfig = None
        self._parameterConfig = None
        self._FMConfig = None
        self._ODEConfig = None
        self._AdjustConfig = None
        self._InterfaceConfig = None
        self._FrameConfig = None
        pass

    def configure(self, accConfig, solverConfig: SolverConfig, AdjustOrbitConfig: AdjustOrbitConfig,
                  ODEConfig:ODEConfig, FMConfig: ForceModelConfig, parameterConfig: ParameterConfig,
                  InterfaceConfig:InterfaceConfig, FrameConfig:FrameConfig):
        self._accConfig = accConfig
        self._AdjustConfig = AdjustOrbitConfig
        self._InterfaceConfig = InterfaceConfig
        self._solverConfig = solverConfig
        self._parameterConfig = parameterConfig
        self._FMConfig = FMConfig
        self._ODEConfig = ODEConfig
        self._FrameConfig = FrameConfig
        self._res = self._solverConfig.DesignMatrixTemp

        '''config adjust orbit'''
        self._RangeRateConfig = self._AdjustConfig.RangeRate()
        self._RangeRateConfig.__dict__.update(self._AdjustConfig.RangeRateConfig.copy())
        self._PathfileConfig = self._AdjustConfig.PathOfFiles()
        self._PathfileConfig.__dict__.update(self._AdjustConfig.PathOfFilesConfig.copy())

        kbr_arc = self._RangeRateConfig.ArcLength
        self.__rms_times = self._RangeRateConfig.OutlierTimes
        self.__upper_RMS = self._RangeRateConfig.OutlierLimit
        self.__kbr_fit = self._RangeRateConfig.ParameterFitting
        self.__iterations = self._RangeRateConfig.Iterations
        '''the arc length to Calibrate kbr: unit [hour]'''
        self.__kbr_arc = kbr_arc * 3600

        return self

    def orbit_integration(self):
        adjustDM = AdjustDM(date_span=self.__data_span, sat=self.__sat)
        adjustDM.configure(SerDer=Orbit_var_2nd_diff, arcNo=self._arcNo, ParameterConfig=self._parameterConfig,
                           AdjustConfig=self._AdjustConfig,
                           FMConfig=self._FMConfig, SolverConfig=self._solverConfig,
                           ODEConfig=self._ODEConfig, InterfaceConfig=self._InterfaceConfig,
                           FrameConfig=self._FrameConfig)
        '''define the directory of res output'''
        res_dir = pathlib.Path(self._res).joinpath(self.__data_span[0] + '_' + self.__data_span[1])
        os.makedirs(res_dir, exist_ok=True)
        # kwargs = {'ChangeTempPath': res_dir}
        adjustDM.calibrate()

        return self

    def get_orbit_design_matrix(self):

        DM_orbit = self.__load_state_vec()
        pp = Parameterization().configure(parameterConfig=self._parameterConfig)

        dm_r = DM_orbit['r']
        dm_v = DM_orbit['v']
        r, v, local_r, global_r, local_v, global_v = [], [], [], [], [], []

        r, local_r, global_r = pp.split_global_local(dm_r, isIncludeStateVec=True)
        v, local_v, global_v = pp.split_global_local(dm_v, isIncludeStateVec=True)

        res_dir = self._res + '/' + self.__data_span[0] + '_' + self.__data_span[1]
        res_filename = res_dir + '/Orbit_' + str(self._arcNo) + '_' + self._satName + '.hdf5'

        h5 = h5py.File(res_filename, 'w')
        h5.create_dataset('t', data=DM_orbit['t'])
        h5.create_dataset('r', data=r)
        h5.create_dataset('v', data=v)
        h5.create_dataset('local_r', data=local_r)
        h5.create_dataset('global_r', data=global_r)
        h5.create_dataset('local_v', data=local_v)
        h5.create_dataset('global_v', data=global_v)
        h5.close()
        pass

    def get_sst_design_matrix(self):
        """
        Obtain the SST design matrix with the orbit design matrix.
        :return:
        """
        SSTOption = self._solverConfig.SSTOption

        if SSTObserve[SSTOption] == SSTObserve.Range:
            self.__getRangeDM()
        elif SSTObserve[SSTOption] == SSTObserve.RangeRate:
            self.__getRangeRateDM()
        elif SSTObserve[SSTOption] == SSTObserve.RangeAcc:
            self.__getRangeAccDM()

    def __getRangeDM(self):
        pass

    def __getRangeRateDM(self):
        res_dir = self._res + '/' + self.__data_span[0] + '_' + self.__data_span[1]
        res_filename = res_dir + '/Orbit_' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        DM_orbit = h5py.File(res_filename, 'r')
        """load data"""
        self.Time = DM_orbit['t'][:]
        self.__PosA = DM_orbit['r'][:, 0:3]
        self.__PosB = DM_orbit['r'][:, 3:6]
        self.__VelA = DM_orbit['v'][:, 0:3]
        self.__VelB = DM_orbit['v'][:, 3:6]
        # n = self.mean_motion_of_the_mid_point()

        rA, rB, vA, vB = DM_orbit['r'][:, 0:3], DM_orbit['r'][:, 3:6], DM_orbit['v'][:, 0:3], DM_orbit['v'][:, 3:6]

        sst = SSTpartial(rA, vA, rB, vB)
        dd_r1, dd_v1, dd_r2, dd_v2 = sst.getPartial_RangeRate()

        local_1 = dd_r1[:, None, :] @ DM_orbit['local_r'][:, 0:3] + \
                  dd_v1[:, None, :] @ DM_orbit['local_v'][:, 0:3]
        local_2 = dd_r2[:, None, :] @ DM_orbit['local_r'][:, 3:6] + \
                  dd_v2[:, None, :] @ DM_orbit['local_v'][:, 3:6]
        Global = dd_r1[:, None, :] @ DM_orbit['global_r'][:, 0:3] + \
                 dd_v1[:, None, :] @ DM_orbit['global_v'][:, 0:3] + \
                 dd_r2[:, None, :] @ DM_orbit['global_r'][:, 3:6] + \
                 dd_v2[:, None, :] @ DM_orbit['global_v'][:, 3:6]
        # '''divide arcs to Calibrate kbr individually'''
        # node = [self.Time[0]]
        # t1 = self.Time[0]
        # while True:
        #     t2 = t1 + self.__kbr_arc
        #     if np.fabs(t2 - self.Time[-1]) <= 0.5 * 3600:
        #         '''difference less than 0.5 hour, end'''
        #         node.append(t2)
        #         break
        #     elif len(self.Time[self.Time >= t2]) <= 100:
        #         '''the rest points are too short, e.g., less than 100: include the rest into the last arc'''
        #         node[-1] = self.Time[-1] + 10  # +10, a little bigger than the last point to ensure all is included
        #         break
        #     else:
        #         node.append(t2)
        #         t1 = t2
        # arclist_t = []
        # for i in range(len(node) - 1):
        #     index = (self.Time < node[i + 1]) * (self.Time >= node[i])
        #     arclist_t.append(self.Time[index])
        # post_t = np.zeros(0)
        # for i in range(len(arclist_t)):
        #     t = arclist_t[i]
        #     if len(t) == 0:
        #         continue
        #     post_t = np.append(post_t, t)
        # orbit_index = [list(self.Time).index(x) for x in post_t]
        # '''design matrix fit'''
        # for i in range(np.shape(local_1[:, 0, :])[1]):
        #     local_1[:, 0, :][orbit_index, i] = self.__design_fit(obs=local_1[:, 0, :][:, i], n=n, node=node)
        #     local_2[:, 0, :][orbit_index, i] = self.__design_fit(obs=local_2[:, 0, :][:, i], n=n, node=node)
        # for i in range(np.shape(Global[:, 0, :])[1]):
        #     Global[:, 0, :][orbit_index, i] = self.__design_fit(obs=Global[:, 0, :][:, i], n=n, node=node)
        '''make a record'''
        res_dir = self._res + '/' + self.__data_span[0] + '_' + self.__data_span[1]
        res_filename = res_dir + '/' + SSTObserve.RangeRate.name + '_' + str(self._arcNo) + '.hdf5'
        h5 = h5py.File(res_filename, 'w')
        h5.create_dataset('t', data=DM_orbit['t'][:])
        h5.create_dataset('local', data=np.concatenate((local_1[:, 0, :], local_2[:, 0, :]), axis=1))
        h5.create_dataset('global', data=Global[:, 0, :])
        h5.close()
        pass

    def __getRangeAccDM(self):
        pass

    def __load_state_vec(self):
        """
        load orbit (and variational-equation) state vectors
        :return:
        """
        orbit = {}
        res_dir = self._res + '/' + self.__data_span[0] + '_' + self.__data_span[1]
        res_filename = res_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        orbit['r'] = h5['r'][()]
        orbit['v'] = h5['v'][()]
        orbit['t'] = h5['t'][()]
        h5.close()
        return orbit

    def __design_fit(self, obs, n, node):
        index = None
        '''attribute data into each kbr arc'''
        arclist_t, arclist_n, arclist_obs = [], [], []
        for i in range(len(node) - 1):
            index = (self.Time < node[i + 1]) * (self.Time >= node[i])
            arclist_t.append(self.Time[index])
            arclist_n.append(n[index])
            arclist_obs.append(obs[index])

        '''Calibrate kbr arc by arc'''
        post_t, post_residual = np.zeros(0), np.zeros(0)
        dm_list = []
        for i in range(len(arclist_t)):
            t, n, obs = arclist_t[i], arclist_n[i], arclist_obs[i]
            if len(t) == 0:
                continue
            kf = FitSST(t, n, option=self.__kbr_fit)
            dm = kf.get_design_matrix()

            for iter in range(self.__iterations):
                residual = kf.get_residual(dm.copy(), obs)
                obs = residual
            post_t = np.append(post_t, t)
            post_residual = np.append(post_residual, obs)
            dm_list.append(dm)

        return post_residual

    def mean_motion_of_the_mid_point(self):
        mid_pos = (self.__PosA + self.__PosB) / 2
        mid_vel = (self.__VelA + self.__VelB) / 2

        oe = OrbitalElements()
        oe.setState(x=mid_pos[:, 0], y=mid_pos[:, 1], z=mid_pos[:, 2], vx=mid_vel[:, 0], vy=mid_vel[:, 1],
                    vz=mid_vel[:, 2])

        return oe.getMeanMotion()
