from src.Preference.Pre_NEQ import NEQConfig
from src.Preference.EnumType import Payload, SatID, SSTObserve
from src.Interface.ArcOperation import ArcData
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Preference.Pre_Solver import SolverConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Preference.Pre_Interface import InterfaceConfig
from src.Auxilary.Outlier import Outlier
from scipy.linalg import qr, toeplitz, cholesky, eigh, solve_triangular
from scipy.sparse import block_diag
import numpy as np
import h5py
from tqdm import tqdm
import os


class GravityNEQperArc:
    """
    Gravity inversion in terms of spherical harmonic.
    Retrieve the design matrix and observations, then transform the observation-equation into the normal equations.
    """
    def __init__(self, arcNo: int, sat, date_span):
        self.__date_span = date_span
        self.__date_begin = date_span[0]
        self.__date_end = date_span[1]
        self.__arcNo = arcNo
        self.__config = {}
        self._orbitH5 = None
        self._satName = None
        self.__orbit_nominal = None
        self.__t_nominal = None
        self.__local_r = None
        self.__global_r = None
        self._solverConfig = None
        self.__saveData = None
        self._NEQConfig = None
        self._OrbitAConfig = None
        self._OrbitBConfig = None
        self._SSTConfig = None
        self._PathOfFilesConfig = None
        self._AdjustConfig = None
        self._AdjustPathConfig = None
        self._InterfaceConfig = None
        self._rd = None

        self._sat = [key for key, value in sat.items() if value]

        for sat in self._sat:
            if self._satName is None:
                self._satName = sat
            else:
                self._satName = self._satName + sat
        pass

    def configure(self, NEQConfig:NEQConfig, AdjustOrbitConfig: AdjustOrbitConfig,
                  solverConfig: SolverConfig, InterfaceConfig: InterfaceConfig, **kwargs):
        self._NEQConfig = NEQConfig
        self._solverConfig = solverConfig
        self._InterfaceConfig = InterfaceConfig
        '''config adjust config'''
        self._AdjustConfig = AdjustOrbitConfig.Orbit()
        self._AdjustConfig.__dict__.update(AdjustOrbitConfig.OrbitConfig.copy())
        '''config adjust path'''
        self._AdjustPathConfig = AdjustOrbitConfig.PathOfFiles()
        self._AdjustPathConfig.__dict__.update(AdjustOrbitConfig.PathOfFilesConfig.copy())
        '''config NEQ'''
        self._OrbitAConfig = self._NEQConfig.OrbitA()
        self._OrbitAConfig.__dict__.update(self._NEQConfig.OrbitAConfig.copy())
        self._OrbitBConfig = self._NEQConfig.OrbitB()
        self._OrbitBConfig.__dict__.update(self._NEQConfig.OrbitBConfig.copy())
        self._SSTConfig = self._NEQConfig.SST()
        self._SSTConfig.__dict__.update(self._NEQConfig.SSTConfig.copy())
        self._PathOfFilesConfig = self._NEQConfig.PathOfFiles()
        self._PathOfFilesConfig.__dict__.update(self._NEQConfig.PathOfFilesConfig.copy())
        self.__config['satA'] = self._OrbitAConfig
        self.__config['satB'] = self._OrbitBConfig
        self.__config['SST'] = self._SSTConfig

        self._rd = ArcData(interfaceConfig=self._InterfaceConfig)
        res_dir = self._PathOfFilesConfig.NormalEqTemp + '/' + self.__date_begin + '_' + self.__date_end + '/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        self.res_filename = res_dir + str(self.__arcNo) + '.hdf5'
        # self.__saveData = h5py.File(res_filename, 'r')
        # self.__saveData = h5py.File(res_filename, 'w')
        return self

    def get_neq(self):
        """
        three steps to acquire the normal equations. Assure the 'configure' is called before.
        :return:
        """
        res_dir = self._solverConfig.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = res_dir + '/Orbit_' + str(self.__arcNo) + '_' + self._satName + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        self.__orbit_nominal = h5['r'][()]
        self.__t_nominal = h5['t'][()]
        self.__local_r = h5['local_r'][()]
        self.__global_r = h5['global_r'][()]
        h5.close()

        # state_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp). \
        #     joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        # state_filename = state_dir.joinpath(str(self.__arcNo) + '_' + self._satName + '.hdf5')
        # h5 = h5py.File(state_filename, 'r')
        # self.__orbit_nominal = h5['r'][()][:, :, 0]
        # h5.close()
        saveData = h5py.File(self.res_filename, 'w')
        for i in tqdm(range(3), desc='Build normal equations: '):
            if i == 0:
                self.get_neq_orbit(SatID.A, saveData)
            elif i == 1:
                self.get_neq_orbit(SatID.B, saveData)
            elif i == 2:
                self.get_neq_range_rate(saveData)
        saveData.close()
        return self

    def get_neq_orbit(self, sat: SatID, saveData):
        """
        get Normal Equations for the orbit only
        :return:
        """
        if sat == SatID.A:
            config = self.__config['satA']
            orbit_nominal = self.__orbit_nominal[:, 0:3]
            local_r = self.__local_r[:, 0:3, :]
            global_r = self.__global_r[:, 0:3, :]
        else:
            config = self.__config['satB']
            orbit_nominal = self.__orbit_nominal[:, 3:6]
            local_r = self.__local_r[:, 3:6, :]
            global_r = self.__global_r[:, 3:6, :]
        t_nominal = self.__t_nominal
        if not config.IsRequired:
            return None

        sgima_path = self._solverConfig.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1]

        self._rd = ArcData(interfaceConfig=self._InterfaceConfig)

        if config.UseKinematicOrbit:
            kin_orbit = self._rd.getData(arc=self.__arcNo, kind=Payload.KinematicOrbit, sat=sat)
            orbit_obs = kin_orbit[:, 0:4]
            # orbit_obs = self.kin_obs_outlier(sat=sat, kin_orbit=orbit_obs)
        else:
            orbit_obs = self._rd.getData(arc=self.__arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]
        t_obs = orbit_obs[:, 0]
        orbit_obs = orbit_obs[t_obs % 60 == 0, :]
        # kin_obs = kin_orbit[t_obs % 60 == 0, 4:]
        # sigma = self.build_block_diag_matrix(kin_obs)
        t_obs = t_obs[t_obs % 60 == 0]
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]
        '''Reassignment'''
        t_nominal = t_nominal[new_index]
        orbit_nominal = orbit_nominal[new_index, :]
        local_r = local_r[new_index, :, :]
        global_r = global_r[new_index, :, :]
        # orbit_obs = orbit_obs[new_index, :]
        '''Expanding the observations'''
        obs = orbit_obs[:, 1:] - orbit_nominal

        obs = obs.reshape((-1, 1))

        # TODO: Fix the kinematic orbit

        '''Expanding the design matrix'''
        n_local = np.shape(local_r)[-1]
        n_global = np.shape(global_r)[-1]
        local_r = local_r.reshape((-1, n_local))
        global_r = global_r.reshape((-1, n_global))

        # with h5py.File(("orbit_" + sat.name + '_' + str(self.__arcNo) + '.h5'), "w") as f:
        #     f.create_dataset("global_r", data=global_r)
        #     f.create_dataset("local_r", data=local_r)
        #     f.create_dataset("obs", data=obs)
        sigmafilename = sgima_path + '/Orbit_' + str(self.__arcNo) + str(sat.name) + '.hdf5'
        if os.path.exists(sigmafilename):
            sigmah5 = h5py.File(sigmafilename, "r")
            if "Sigma" in sigmah5:
                sigma = sigmah5['Sigma'][:]
                # sigma_y = sigmah5['Sigma_y'][:]
                # sigma_z = sigmah5['Sigma_z'][:]

                index = np.array(new_index)

                idx = np.empty(3*len(index), dtype=int)
                idx[0::3] = 3 * index + 0
                idx[1::3] = 3 * index + 1
                idx[2::3] = 3 * index + 2

                sigma = sigma[np.ix_(idx, idx)]
                # sigma_y = sigma_y[np.ix_(index, index)]
                # sigma_z = sigma_z[np.ix_(index, index)]

                N, l = GeoMathKit.keepGlobal(dm_global=global_r, dm_local=local_r,
                                             obs=obs, p=sigma)
            else:
                N, l = GeoMathKit.keepGlobal(dm_global=global_r, dm_local=local_r, obs=obs, p=None)
            sigmah5.close()
        else:
            N, l = GeoMathKit.keepGlobal(dm_global=global_r, dm_local=local_r, obs=obs, p=None)

        saveData.create_dataset('Orbit_N_%1s' % sat.name, data=N)
        saveData.create_dataset('Orbit_l_%1s' % sat.name, data=l)
        pass

    def get_neq_orbit_V2(self, sat: SatID):
        """
        get Normal Equations for the orbit only
        :return:
        """
        if sat == SatID.A:
            config = self.__config['satA']
            orbit_nominal = self.__orbit_nominal[:, 0:3]
            local_r = self.__local_r[:, 0:3, :]
            global_r = self.__global_r[:, 0:3, :]
        else:
            config = self.__config['satB']
            orbit_nominal = self.__orbit_nominal[:, 3:6]
            local_r = self.__local_r[:, 3:6, :]
            global_r = self.__global_r[:, 3:6, :]
        t_nominal = self.__t_nominal
        if not config.IsRequired:
            return None

        kin_obs = self._rd.getData(arc=self.__arcNo, kind=Payload.KinematicOrbit, sat=sat)[:, 0:4]
        gnv_obs = self._rd.getData(arc=self.__arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]

        kin_outlier = self.get_kin_obs(kin_obs=kin_obs, orbit_nominal=gnv_obs)
        orbit_obs = self.get_orbit_obs(gnv_obs=gnv_obs, Kin_obs=kin_outlier)
        t_obs = orbit_obs[:, 0]
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        commonTime = set(t_obs.astype(np.int64))
        commonTime = commonTime & set(t_nominal.astype(np.int64))
        commonTime = np.array(list(commonTime))
        commonTime.sort()

        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        nominal_index = [list(int_t_nominal).index(x) for x in commonTime]
        obs_index = [list(int_t_obs).index(x) for x in commonTime]
        '''Reassignment'''
        t_nominal = t_nominal[nominal_index]
        orbit_nominal = orbit_nominal[nominal_index, :]
        local_r = local_r[nominal_index, :, :]
        global_r = global_r[nominal_index, :, :]
        orbit_obs = orbit_obs[obs_index, :]
        '''Expanding the observations'''
        obs = orbit_obs[:, 1:] - orbit_nominal
        '''orbit outlier'''
        index = Outlier(rms_times=self._AdjustConfig.OutlierTimes, upper_RMS=self._AdjustConfig.OutlierLimit).remove_V2(
            t_nominal, obs[:, 0])
        '''Reassignment'''
        obs = obs[index, :]
        local_r = local_r[index, :, :]
        global_r = global_r[index, :, :]
        obs = obs.reshape((-1, 1))
        # TODO: Fix the kinematic orbit
        '''Expanding the design matrix'''
        n_global = np.shape(global_r)[-1]
        global_r = global_r.reshape((-1, n_global))
        n_local = np.shape(local_r)[-1]
        local_r = local_r.reshape((-1, n_local))

        N, l = GeoMathKit.keepGlobal2(dm_global=global_r, dm_local=local_r, obs=obs)

        self.__saveData.create_dataset('Orbit_N_%1s' % sat.name, data=N)
        self.__saveData.create_dataset('Orbit_l_%1s' % sat.name, data=l)
        pass

    def get_neq_range_rate(self, saveData):
        """
        get Normal Equations for the range-rate only
        :return:
        """
        config = self.__config['SST']

        if not config.IsRequired:
            return None

        assert config.measurement == SSTObserve.RangeRate.value, 'Range-rate is not allowed by user setting'
        sgima_path = self._solverConfig.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1]

        dm_dir = self._AdjustPathConfig.RangeRateTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = dm_dir + '/' + SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        t_obs = h5['post_t'][()]
        obs = h5['post_residual'][()]
        local_dm_empirical = h5['design_matrix'][()]
        h5.close()

        dm_dir = self._solverConfig.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = dm_dir + '/' + SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        t_dm = h5['t'][()]
        global_dm = h5['global'][()]
        local_dm = h5['local'][()]
        h5.close()

        int_t_obs = t_obs.astype(np.int64)
        int_t_dm = t_dm.astype(np.int64)
        new_index = [list(int_t_dm).index(x) for x in int_t_obs]

        '''Reassignment'''
        t_dm = t_dm[new_index]
        global_dm = global_dm[new_index, :]
        local_dm = local_dm[new_index, :]

        '''Combination'''
        if config.CalibrateEmpiricalParameters:
            local_dm = np.concatenate((local_dm_empirical, local_dm), axis=1)
        
        # with h5py.File(("kbrr_" + str(self.__arcNo) + '.h5'), "w") as f:
        #     f.create_dataset("global_dm", data=global_dm)
        #     f.create_dataset("local_dm", data=local_dm)
        #     f.create_dataset("obs", data=obs)
        obs = obs.reshape((-1, 1))

        sigmafilename = sgima_path + '/' + SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5'
        if os.path.exists(sigmafilename):
            sigmah5 = h5py.File(sigmafilename, "r")
            if "Sigma" in sigmah5:
                index = np.array(new_index, dtype=int)

                sigma = sigmah5['Sigma'][:]
                sigma = sigma[np.ix_(index, index)]

                N, l = GeoMathKit.keepGlobal(dm_global=global_dm, dm_local=local_dm, obs=obs, p=sigma)
            else:
                N, l = GeoMathKit.keepGlobal(dm_global=global_dm, dm_local=local_dm, obs=obs, p=None)
            sigmah5.close()
        else:
            N, l = GeoMathKit.keepGlobal(dm_global=global_dm, dm_local=local_dm, obs=obs, p=None)

        saveData.create_dataset('%s_N' % SSTObserve.RangeRate.name, data=N)
        saveData.create_dataset('%s_l' % SSTObserve.RangeRate.name, data=l)

        pass

    def get_kin_obs(self,kin_obs, orbit_nominal):
        t_nominal = orbit_nominal[:, 0]
        t_obs = kin_obs[:, 0]
        if len(t_obs) == 0:
            return kin_obs
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]
        '''Reassignment'''
        t_nominal = t_nominal[new_index]
        orbit_nominal = orbit_nominal[new_index, :]
        '''Outlier'''
        before_obs = kin_obs[:, 1:] - orbit_nominal[:, 1:]
        before_obs = np.sum(before_obs, axis=1)
        index = Outlier(rms_times=self._AdjustConfig.OutlierTimes, upper_RMS=self._AdjustConfig.OutlierLimit).remove_V2(
            t_nominal, before_obs)
        return kin_obs[index, :]

    def get_orbit_obs(self, gnv_obs, Kin_obs):
        GNV_obs = gnv_obs
        t_nominal = GNV_obs[:, 0]
        t_obs = Kin_obs[:, 0]
        if len(t_obs) == 0:
            return GNV_obs
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]

        GNV_obs[new_index, :] = Kin_obs

        return GNV_obs

    def rms(self, x: np.ndarray):
        """
        :param x: 1-dim
        :return:
        """
        # 确认公式
        return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])

    def kin_obs_outlier(self, kin_orbit, sat):
        gnv_orbit = self._rd.getData(arc=self.__arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]
        gnv_t = gnv_orbit[:, 0]
        kin_t = kin_orbit[:, 0]
        if len(kin_t) == 0:
            return gnv_orbit
        int_gnv_t = gnv_t.astype(np.int64)
        int_kin_t = kin_t.astype(np.int64)

        new_index = [list(int_gnv_t).index(x) for x in int_kin_t]
        gnv_orbit = gnv_orbit[new_index, :]

        '''Outlier'''
        before_obs = kin_orbit[:, 1:] - gnv_orbit[:, 1:]
        before_obs = np.sum(before_obs, axis=1)
        index = Outlier(rms_times=self._AdjustConfig.OutlierTimes, upper_RMS=self._AdjustConfig.OutlierLimit).remove_V2(
            kin_t, before_obs)

        kin_orbit = kin_orbit[index, :]

        return kin_orbit

    def get_kin_cov(self):
        pass

    def build_block_diag_matrix(self, data):
        """
        Parameters
        ----------
        data : ndarray (N,6)
            每个历元 [xx, yy, zz, xy, xz, yz]

        Returns
        -------
        ndarray (3N, 3N)
            稠密块对角矩阵
        """

        data = np.asarray(data)
        N = data.shape[0]

        # 初始化大矩阵
        M = np.zeros((3 * N, 3 * N), dtype=data.dtype)

        # 每个块的起始索引
        idx = np.arange(N) * 3

        # 对角元
        M[idx, idx] = data[:, 0]  # xx
        M[idx + 1, idx + 1] = data[:, 1]  # yy
        M[idx + 2, idx + 2] = data[:, 2]  # zz

        # 非对角元
        M[idx, idx + 1] = data[:, 3]  # xy
        M[idx + 1, idx] = data[:, 3]

        M[idx, idx + 2] = data[:, 4]  # xz
        M[idx + 2, idx] = data[:, 4]

        M[idx + 1, idx + 2] = data[:, 5]  # yz
        M[idx + 2, idx + 1] = data[:, 5]

        return M


class GravityNEQperArc_v2:
    """
    Gravity inversion in terms of spherical harmonic.
    Retrieve the design matrix and observations, then transform the observation-equation into the normal equations.
    """

    def __init__(self, arcNo: int, sat, date_span):
        self.__date_span = date_span
        self.__date_begin = date_span[0]
        self.__date_end = date_span[1]
        self.__arcNo = arcNo
        self.__config = {}
        self._orbitH5 = None
        self._satName = None
        self.__orbit_nominal = None
        self.__t_nominal = None
        self.__local_r = None
        self.__global_r = None
        self._solverConfig = None
        self.__saveData = None
        self._NEQConfig = None
        self._OrbitAConfig = None
        self._OrbitBConfig = None
        self._SSTConfig = None
        self._PathOfFilesConfig = None
        self._AdjustConfig = None
        self._AdjustPathConfig = None
        self._InterfaceConfig = None
        self._rd = None

        self._sat = [key for key, value in sat.items() if value]

        for sat in self._sat:
            if self._satName is None:
                self._satName = sat
            else:
                self._satName = self._satName + sat
        pass

    def configure(self, NEQConfig: NEQConfig, AdjustOrbitConfig: AdjustOrbitConfig,
                  solverConfig: SolverConfig, InterfaceConfig: InterfaceConfig, **kwargs):
        self._NEQConfig = NEQConfig
        self._solverConfig = solverConfig
        self._InterfaceConfig = InterfaceConfig
        '''config adjust config'''
        self._AdjustConfig = AdjustOrbitConfig.Orbit()
        self._AdjustConfig.__dict__.update(AdjustOrbitConfig.OrbitConfig.copy())
        '''config adjust path'''
        self._AdjustPathConfig = AdjustOrbitConfig.PathOfFiles()
        self._AdjustPathConfig.__dict__.update(AdjustOrbitConfig.PathOfFilesConfig.copy())
        '''config NEQ'''
        self._OrbitAConfig = self._NEQConfig.OrbitA()
        self._OrbitAConfig.__dict__.update(self._NEQConfig.OrbitAConfig.copy())
        self._OrbitBConfig = self._NEQConfig.OrbitB()
        self._OrbitBConfig.__dict__.update(self._NEQConfig.OrbitBConfig.copy())
        self._SSTConfig = self._NEQConfig.SST()
        self._SSTConfig.__dict__.update(self._NEQConfig.SSTConfig.copy())
        self._PathOfFilesConfig = self._NEQConfig.PathOfFiles()
        self._PathOfFilesConfig.__dict__.update(self._NEQConfig.PathOfFilesConfig.copy())
        self.__config['satA'] = self._OrbitAConfig
        self.__config['satB'] = self._OrbitBConfig
        self.__config['SST'] = self._SSTConfig

        self._rd = ArcData(interfaceConfig=self._InterfaceConfig)
        res_dir = self._PathOfFilesConfig.NormalEqTemp + '/' + self.__date_begin + '_' + self.__date_end + '/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        self.res_filename = res_dir + str(self.__arcNo) + '.hdf5'
        # self.__saveData = h5py.File(res_filename, 'r')
        # self.__saveData = h5py.File(res_filename, 'w')
        return self

    def get_neq(self):
        """
        three steps to acquire the normal equations. Assure the 'configure' is called before.
        :return:
        """
        res_dir = self._solverConfig.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = res_dir + '/Orbit_' + str(self.__arcNo) + '_' + self._satName + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        self.__orbit_nominal = h5['r'][()]
        self.__t_nominal = h5['t'][()]
        self.__local_r = h5['local_r'][()]
        self.__global_r = h5['global_r'][()]
        h5.close()

        # state_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp). \
        #     joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        # state_filename = state_dir.joinpath(str(self.__arcNo) + '_' + self._satName + '.hdf5')
        # h5 = h5py.File(state_filename, 'r')
        # self.__orbit_nominal = h5['r'][()][:, :, 0]
        # h5.close()
        saveData = h5py.File(self.res_filename, 'w')
        designData = h5py.File(res_filename, 'r+')
        for i in tqdm(range(3), desc='Build normal equations: '):
            if i == 0:
                self.get_neq_orbit(SatID.A, saveData, designData)
            elif i == 1:
                self.get_neq_orbit(SatID.B, saveData, designData)
            elif i == 2:
                self.get_neq_range_rate(saveData)
        saveData.close()
        designData.close()
        return self

    def get_neq_orbit(self, sat: SatID, saveData, designData):
        """
        get Normal Equations for the orbit only
        :return:
        """
        if sat == SatID.A:
            config = self.__config['satA']
            orbit_nominal = self.__orbit_nominal[:, 0:3]
            local_r = self.__local_r[:, 0:3, :]
            global_r = self.__global_r[:, 0:3, :]
        else:
            config = self.__config['satB']
            orbit_nominal = self.__orbit_nominal[:, 3:6]
            local_r = self.__local_r[:, 3:6, :]
            global_r = self.__global_r[:, 3:6, :]
        t_nominal = self.__t_nominal
        if not config.IsRequired:
            return None

        self._rd = ArcData(interfaceConfig=self._InterfaceConfig)

        if config.UseKinematicOrbit:
            kin_orbit = self._rd.getData(arc=self.__arcNo, kind=Payload.KinematicOrbit, sat=sat)
            orbit_obs = kin_orbit[:, 0:4]
            # orbit_obs = self.kin_obs_outlier(sat=sat, kin_orbit=orbit_obs)
        else:
            orbit_obs = self._rd.getData(arc=self.__arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]
        t_obs = orbit_obs[:, 0]
        # orbit_obs = orbit_obs[t_obs % 60 == 0, :]
        kin_obs = kin_orbit[:, 4:]
        sigma = self.build_block_diag_matrix(kin_obs)
        # t_obs = t_obs[t_obs % 60 == 0]
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]
        '''Reassignment'''
        t_nominal = t_nominal[new_index]
        orbit_nominal = orbit_nominal[new_index, :]
        local_r = local_r[new_index, :, :]
        global_r = global_r[new_index, :, :]
        # orbit_obs = orbit_obs[new_index, :]
        '''Expanding the observations'''
        obs = orbit_obs[:, 1:] - orbit_nominal

        obs = obs.reshape((-1, 1))

        # TODO: Fix the kinematic orbit

        '''Expanding the design matrix'''
        n_local = np.shape(local_r)[-1]
        n_global = np.shape(global_r)[-1]
        local_r = local_r.reshape((-1, n_local))
        global_r = global_r.reshape((-1, n_global))

        # if p is None:
        #     p = np.eye(np.shape(global_r.T)[-1])
        # I = np.eye(np.shape(local_r.T)[-1])
        '''Cholesky白化'''
        # sigma = np.eye(np.shape(global_r.T)[-1])
        L = cholesky(sigma, lower=True)
        Agw = solve_triangular(L, global_r, lower=True)  # n x p
        Alw = solve_triangular(L, local_r, lower=True)  # n x q
        lw = solve_triangular(L, obs, lower=True)

        if 'newindex' in designData:
            del designData["newindex"]
        if 'Orbit_t_%1s' % sat.name in designData:
            del designData['Orbit_t_%1s' % sat.name]
        if 'Orbit_global_%1s' % sat.name in designData:
            del designData['Orbit_global_%1s' % sat.name]
        if 'Orbit_local_%1s' % sat.name in designData:
            del designData['Orbit_local_%1s' % sat.name]
        if 'Orbit_obs_%1s' % sat.name in designData:
            del designData['Orbit_obs_%1s' % sat.name]

        designData.create_dataset('newindex', data=new_index)
        designData.create_dataset('Orbit_t_%1s' % sat.name, data=t_nominal)
        designData.create_dataset('Orbit_global_%1s' % sat.name, data=Agw)
        designData.create_dataset('Orbit_local_%1s' % sat.name, data=Alw)
        designData.create_dataset('Orbit_obs_%1s' % sat.name, data=lw)

        N, l = GeoMathKit.keepGlobal_v2(dm_global=Agw, dm_local=Alw, obs=lw)

        saveData.create_dataset('Orbit_N_%1s' % sat.name, data=N)
        saveData.create_dataset('Orbit_l_%1s' % sat.name, data=l)
        pass

    def get_neq_range_rate(self, saveData):
        """
        get Normal Equations for the range-rate only
        :return:
        """
        config = self.__config['SST']

        if not config.IsRequired:
            return None

        assert config.measurement == SSTObserve.RangeRate.value, 'Range-rate is not allowed by user setting'
        sgima_path = self._solverConfig.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1]

        dm_dir = self._AdjustPathConfig.RangeRateTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = dm_dir + '/' + SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        t_obs = h5['post_t'][()]
        obs = h5['post_residual'][()]
        local_dm_empirical = h5['design_matrix'][()]
        h5.close()

        dm_dir = self._solverConfig.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = dm_dir + '/' + SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        t_dm = h5['t'][()]
        global_dm = h5['global'][()]
        local_dm = h5['local'][()]
        h5.close()

        int_t_obs = t_obs.astype(np.int64)
        int_t_dm = t_dm.astype(np.int64)
        new_index = [list(int_t_dm).index(x) for x in int_t_obs]

        '''Reassignment'''
        t_dm = t_dm[new_index]
        global_dm = global_dm[new_index, :]
        local_dm = local_dm[new_index, :]

        '''Combination'''
        if config.CalibrateEmpiricalParameters:
            local_dm = np.concatenate((local_dm_empirical, local_dm), axis=1)

        # with h5py.File(("kbrr_" + str(self.__arcNo) + '.h5'), "w") as f:
        #     f.create_dataset("global_dm", data=global_dm)
        #     f.create_dataset("local_dm", data=local_dm)
        #     f.create_dataset("obs", data=obs)
        obs = obs.reshape((-1, 1))

        '''Cholesky白化'''
        sigma = np.eye(np.shape(global_dm.T)[-1])
        L = cholesky(sigma, lower=True)
        Agw = solve_triangular(L, global_dm, lower=True)  # n x p
        Alw = solve_triangular(L, local_dm, lower=True)  # n x q
        lw = solve_triangular(L, obs, lower=True)

        designkbrr = h5py.File(res_filename, 'r+')
        if '%s_t' % SSTObserve.RangeRate.name in designkbrr:
            del designkbrr['%s_t' % SSTObserve.RangeRate.name]
        if 'newindex' in designkbrr:
            del designkbrr['newindex']
        if '%s_global' % SSTObserve.RangeRate.name in designkbrr:
            del designkbrr['%s_global' % SSTObserve.RangeRate.name]
        if '%s_local' % SSTObserve.RangeRate.name in designkbrr:
            del designkbrr['%s_local' % SSTObserve.RangeRate.name]
        if '%s_obs' % SSTObserve.RangeRate.name in designkbrr:
            del designkbrr['%s_obs' % SSTObserve.RangeRate.name]

        designkbrr.create_dataset('%s_t' % SSTObserve.RangeRate.name, data=t_dm)
        designkbrr.create_dataset('newindex', data=new_index)
        designkbrr.create_dataset('%s_global' % SSTObserve.RangeRate.name, data=Agw)
        designkbrr.create_dataset('%s_local' % SSTObserve.RangeRate.name, data=Alw)
        designkbrr.create_dataset('%s_obs' % SSTObserve.RangeRate.name, data=lw)
        designkbrr.close()

        N, l = GeoMathKit.keepGlobal_v2(dm_global=Agw, dm_local=Alw, obs=lw)

        saveData.create_dataset('%s_N' % SSTObserve.RangeRate.name, data=N)
        saveData.create_dataset('%s_l' % SSTObserve.RangeRate.name, data=l)

        pass

    def get_kin_obs(self, kin_obs, orbit_nominal):
        t_nominal = orbit_nominal[:, 0]
        t_obs = kin_obs[:, 0]
        if len(t_obs) == 0:
            return kin_obs
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]
        '''Reassignment'''
        t_nominal = t_nominal[new_index]
        orbit_nominal = orbit_nominal[new_index, :]
        '''Outlier'''
        before_obs = kin_obs[:, 1:] - orbit_nominal[:, 1:]
        before_obs = np.sum(before_obs, axis=1)
        index = Outlier(rms_times=self._AdjustConfig.OutlierTimes, upper_RMS=self._AdjustConfig.OutlierLimit).remove_V2(
            t_nominal, before_obs)
        return kin_obs[index, :]

    def get_orbit_obs(self, gnv_obs, Kin_obs):
        GNV_obs = gnv_obs
        t_nominal = GNV_obs[:, 0]
        t_obs = Kin_obs[:, 0]
        if len(t_obs) == 0:
            return GNV_obs
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]

        GNV_obs[new_index, :] = Kin_obs

        return GNV_obs

    def rms(self, x: np.ndarray):
        """
        :param x: 1-dim
        :return:
        """
        # 确认公式
        return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])

    def kin_obs_outlier(self, kin_orbit, sat):
        gnv_orbit = self._rd.getData(arc=self.__arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]
        gnv_t = gnv_orbit[:, 0]
        kin_t = kin_orbit[:, 0]
        if len(kin_t) == 0:
            return gnv_orbit
        int_gnv_t = gnv_t.astype(np.int64)
        int_kin_t = kin_t.astype(np.int64)

        new_index = [list(int_gnv_t).index(x) for x in int_kin_t]
        gnv_orbit = gnv_orbit[new_index, :]

        '''Outlier'''
        before_obs = kin_orbit[:, 1:] - gnv_orbit[:, 1:]
        before_obs = np.sum(before_obs, axis=1)
        index = Outlier(rms_times=self._AdjustConfig.OutlierTimes, upper_RMS=self._AdjustConfig.OutlierLimit).remove_V2(
            kin_t, before_obs)

        kin_orbit = kin_orbit[index, :]

        return kin_orbit

    def get_kin_cov(self):
        pass

    def build_block_diag_matrix(self, data):
        """
        Parameters
        ----------
        data : ndarray (N,6)
            每个历元 [xx, yy, zz, xy, xz, yz]

        Returns
        -------
        ndarray (3N, 3N)
            稠密块对角矩阵
        """

        data = np.asarray(data)
        N = data.shape[0]

        # 初始化大矩阵
        M = np.zeros((3 * N, 3 * N), dtype=data.dtype)

        # 每个块的起始索引
        idx = np.arange(N) * 3

        # 对角元
        M[idx, idx] = data[:, 0]  # xx
        M[idx + 1, idx + 1] = data[:, 1]  # yy
        M[idx + 2, idx + 2] = data[:, 2]  # zz

        # 非对角元
        M[idx, idx + 1] = data[:, 3]  # xy
        M[idx + 1, idx] = data[:, 3]

        M[idx, idx + 2] = data[:, 4]  # xz
        M[idx + 2, idx] = data[:, 4]

        M[idx + 1, idx + 2] = data[:, 5]  # yz
        M[idx + 2, idx + 1] = data[:, 5]

        return M


class GravityNEQperArc_v3:
    """
    Gravity inversion in terms of spherical harmonic.
    Retrieve the design matrix and observations, then transform the observation-equation into the normal equations.
    """

    def __init__(self, arcNo: int, sat, date_span):
        self.__date_span = date_span
        self.__date_begin = date_span[0]
        self.__date_end = date_span[1]
        self.__arcNo = arcNo
        self.__config = {}
        self._orbitH5 = None
        self._satName = None
        self.__orbit_nominal = None
        self.__t_nominal = None
        self.__local_r = None
        self.__global_r = None
        self._solverConfig = None
        self.__saveData = None
        self._NEQConfig = None
        self._OrbitAConfig = None
        self._OrbitBConfig = None
        self._SSTConfig = None
        self._PathOfFilesConfig = None
        self._AdjustConfig = None
        self._AdjustPathConfig = None
        self._InterfaceConfig = None
        self._rd = None

        self._sat = [key for key, value in sat.items() if value]

        for sat in self._sat:
            if self._satName is None:
                self._satName = sat
            else:
                self._satName = self._satName + sat
        pass

    def configure(self, NEQConfig: NEQConfig, AdjustOrbitConfig: AdjustOrbitConfig,
                  solverConfig: SolverConfig, InterfaceConfig: InterfaceConfig, **kwargs):
        self._NEQConfig = NEQConfig
        self._solverConfig = solverConfig
        self._InterfaceConfig = InterfaceConfig
        '''config adjust config'''
        self._AdjustConfig = AdjustOrbitConfig.Orbit()
        self._AdjustConfig.__dict__.update(AdjustOrbitConfig.OrbitConfig.copy())
        '''config adjust path'''
        self._AdjustPathConfig = AdjustOrbitConfig.PathOfFiles()
        self._AdjustPathConfig.__dict__.update(AdjustOrbitConfig.PathOfFilesConfig.copy())
        '''config NEQ'''
        self._OrbitAConfig = self._NEQConfig.OrbitA()
        self._OrbitAConfig.__dict__.update(self._NEQConfig.OrbitAConfig.copy())
        self._OrbitBConfig = self._NEQConfig.OrbitB()
        self._OrbitBConfig.__dict__.update(self._NEQConfig.OrbitBConfig.copy())
        self._SSTConfig = self._NEQConfig.SST()
        self._SSTConfig.__dict__.update(self._NEQConfig.SSTConfig.copy())
        self._PathOfFilesConfig = self._NEQConfig.PathOfFiles()
        self._PathOfFilesConfig.__dict__.update(self._NEQConfig.PathOfFilesConfig.copy())
        self.__config['satA'] = self._OrbitAConfig
        self.__config['satB'] = self._OrbitBConfig
        self.__config['SST'] = self._SSTConfig

        self._rd = ArcData(interfaceConfig=self._InterfaceConfig)
        res_dir = self._PathOfFilesConfig.NormalEqTemp + '/' + self.__date_begin + '_' + self.__date_end + '/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        self.res_filename = res_dir + str(self.__arcNo) + '.hdf5'
        # self.__saveData = h5py.File(res_filename, 'r')
        # self.__saveData = h5py.File(res_filename, 'w')
        return self

    def get_neq(self):
        """
        three steps to acquire the normal equations. Assure the 'configure' is called before.
        :return:
        """
        res_dir = self._solverConfig.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = res_dir + '/Orbit_' + str(self.__arcNo) + '_' + self._satName + '.hdf5'
        # state_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp). \
        #     joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        # state_filename = state_dir.joinpath(str(self.__arcNo) + '_' + self._satName + '.hdf5')
        # h5 = h5py.File(state_filename, 'r')
        # self.__orbit_nominal = h5['r'][()][:, :, 0]
        # h5.close()
        saveData = h5py.File(self.res_filename, 'w')
        designData = h5py.File(res_filename, 'r+')
        for i in tqdm(range(3), desc='Build normal equations: '):
            if i == 0:
                self.get_neq_orbit(SatID.A, saveData, designData)
            elif i == 1:
                self.get_neq_orbit(SatID.B, saveData, designData)
            elif i == 2:
                self.get_neq_range_rate(saveData)
        saveData.close()
        designData.close()
        return self

    def get_neq_orbit(self, sat: SatID, saveData, designData):
        """
        get Normal Equations for the orbit only
        :return:
        """
        if sat == SatID.A:
            config = self.__config['satA']
        else:
            config = self.__config['satB']

        t_nominal = designData['t'][()]
        obs = designData['Orbit_obs_%1s' % sat.name][()]
        local_r = designData['Orbit_local_%1s' % sat.name][()]
        global_r = designData['Orbit_global_%1s' % sat.name][()]

        self._rd = ArcData(interfaceConfig=self._InterfaceConfig)

        if config.UseKinematicOrbit:
            kin_orbit = self._rd.getData(arc=self.__arcNo, kind=Payload.KinematicOrbit, sat=sat)
            orbit_obs = kin_orbit[:, 0:4]
            # orbit_obs = self.kin_obs_outlier(sat=sat, kin_orbit=orbit_obs)
        else:
            orbit_obs = self._rd.getData(arc=self.__arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]
        t_obs = orbit_obs[:, 0]
        # orbit_obs = orbit_obs[t_obs % 60 == 0, :]
        # t_obs = t_obs[t_obs % 60 == 0]
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]

        if not config.IsRequired:
            return None

        sgima_path = self._solverConfig.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1]
        sigmafilename = sgima_path + '/Orbit_' + str(self.__arcNo) + str(sat.name) + '.hdf5'
        sigmah5 = h5py.File(sigmafilename, "r")

        sigma = sigmah5['Sigma'][:]
        index = np.array(new_index)
        idx = np.empty(3 * len(index), dtype=int)
        idx[0::3] = 3 * index + 0
        idx[1::3] = 3 * index + 1
        idx[2::3] = 3 * index + 2
        sigma = sigma[np.ix_(idx, idx)]

        '''Cholesky白化'''
        L = cholesky(sigma, lower=True)
        Agw = solve_triangular(L, global_r, lower=True)  # n x p
        Alw = solve_triangular(L, local_r, lower=True)  # n x q
        lw = solve_triangular(L, obs, lower=True)

        if 'Orbit_global_%1s' % sat.name in designData:
            del designData['Orbit_global_%1s' % sat.name]
        if 'Orbit_local_%1s' % sat.name in designData:
            del designData['Orbit_local_%1s' % sat.name]
        if 'Orbit_obs_%1s' % sat.name in designData:
            del designData['Orbit_obs_%1s' % sat.name]

        designData.create_dataset('Orbit_global_%1s' % sat.name, data=Agw)
        designData.create_dataset('Orbit_local_%1s' % sat.name, data=Alw)
        designData.create_dataset('Orbit_obs_%1s' % sat.name, data=lw)

        N, l = GeoMathKit.keepGlobal_v2(dm_global=Agw, dm_local=Alw, obs=lw)

        saveData.create_dataset('Orbit_N_%1s' % sat.name, data=N)
        saveData.create_dataset('Orbit_l_%1s' % sat.name, data=l)
        pass

    def get_neq_range_rate(self, saveData):
        """
        get Normal Equations for the range-rate only
        :return:
        """
        config = self.__config['SST']

        if not config.IsRequired:
            return None

        assert config.measurement == SSTObserve.RangeRate.value, 'Range-rate is not allowed by user setting'
        sgima_path = self._solverConfig.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1]

        dm_dir = self._solverConfig.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = dm_dir + '/' + SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5'

        designData = h5py.File(res_filename, 'r+')
        t_dm = designData['t'][()]
        new_index = designData['newindex'][()]
        global_dm = designData['%s_global' % SSTObserve.RangeRate.name][()]
        local_dm = designData['%s_local' % SSTObserve.RangeRate.name][()]
        obs = designData['%s_obs' % SSTObserve.RangeRate.name][()]

        sigmafilename = sgima_path + '/' + SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5'
        sigmah5 = h5py.File(sigmafilename, "r")
        index = np.array(new_index, dtype=int)
        sigma = sigmah5['Sigma'][:]
        sigma = sigma[np.ix_(index, index)]
        '''Cholesky白化'''

        L = cholesky(sigma, lower=True)
        Agw = solve_triangular(L, global_dm, lower=True)  # n x p
        Alw = solve_triangular(L, local_dm, lower=True)  # n x q
        lw = solve_triangular(L, obs, lower=True)

        if '%s_global' % SSTObserve.RangeRate.name in designData:
            del designData['%s_global' % SSTObserve.RangeRate.name]
        if '%s_local' % SSTObserve.RangeRate.name in designData:
            del designData['%s_local' % SSTObserve.RangeRate.name]
        if '%s_obs' % SSTObserve.RangeRate.name in designData:
            del designData['%s_obs' % SSTObserve.RangeRate.name]

        designData.create_dataset('%s_global' % SSTObserve.RangeRate.name, data=Agw)
        designData.create_dataset('%s_local' % SSTObserve.RangeRate.name, data=Alw)
        designData.create_dataset('%s_obs' % SSTObserve.RangeRate.name, data=lw)

        designData.close()

        N, l = GeoMathKit.keepGlobal_v2(dm_global=Agw, dm_local=Alw, obs=lw)

        saveData.create_dataset('%s_N' % SSTObserve.RangeRate.name, data=N)
        saveData.create_dataset('%s_l' % SSTObserve.RangeRate.name, data=l)

        pass

    def get_kin_obs(self, kin_obs, orbit_nominal):
        t_nominal = orbit_nominal[:, 0]
        t_obs = kin_obs[:, 0]
        if len(t_obs) == 0:
            return kin_obs
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]
        '''Reassignment'''
        t_nominal = t_nominal[new_index]
        orbit_nominal = orbit_nominal[new_index, :]
        '''Outlier'''
        before_obs = kin_obs[:, 1:] - orbit_nominal[:, 1:]
        before_obs = np.sum(before_obs, axis=1)
        index = Outlier(rms_times=self._AdjustConfig.OutlierTimes, upper_RMS=self._AdjustConfig.OutlierLimit).remove_V2(
            t_nominal, before_obs)
        return kin_obs[index, :]

    def get_orbit_obs(self, gnv_obs, Kin_obs):
        GNV_obs = gnv_obs
        t_nominal = GNV_obs[:, 0]
        t_obs = Kin_obs[:, 0]
        if len(t_obs) == 0:
            return GNV_obs
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''
        int_t_obs = t_obs.astype(np.int64)
        int_t_nominal = t_nominal.astype(np.int64)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]

        GNV_obs[new_index, :] = Kin_obs

        return GNV_obs

    def rms(self, x: np.ndarray):
        """
        :param x: 1-dim
        :return:
        """
        # 确认公式
        return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])

    def kin_obs_outlier(self, kin_orbit, sat):
        gnv_orbit = self._rd.getData(arc=self.__arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]
        gnv_t = gnv_orbit[:, 0]
        kin_t = kin_orbit[:, 0]
        if len(kin_t) == 0:
            return gnv_orbit
        int_gnv_t = gnv_t.astype(np.int64)
        int_kin_t = kin_t.astype(np.int64)

        new_index = [list(int_gnv_t).index(x) for x in int_kin_t]
        gnv_orbit = gnv_orbit[new_index, :]

        '''Outlier'''
        before_obs = kin_orbit[:, 1:] - gnv_orbit[:, 1:]
        before_obs = np.sum(before_obs, axis=1)
        index = Outlier(rms_times=self._AdjustConfig.OutlierTimes, upper_RMS=self._AdjustConfig.OutlierLimit).remove_V2(
            kin_t, before_obs)

        kin_orbit = kin_orbit[index, :]

        return kin_orbit

    def get_kin_cov(self):
        pass

    def build_block_diag_matrix(self, data):
        """
        Parameters
        ----------
        data : ndarray (N,6)
            每个历元 [xx, yy, zz, xy, xz, yz]

        Returns
        -------
        ndarray (3N, 3N)
            稠密块对角矩阵
        """

        data = np.asarray(data)
        N = data.shape[0]

        # 初始化大矩阵
        M = np.zeros((3 * N, 3 * N), dtype=data.dtype)

        # 每个块的起始索引
        idx = np.arange(N) * 3

        # 对角元
        M[idx, idx] = data[:, 0]  # xx
        M[idx + 1, idx + 1] = data[:, 1]  # yy
        M[idx + 2, idx + 2] = data[:, 2]  # zz

        # 非对角元
        M[idx, idx + 1] = data[:, 3]  # xy
        M[idx + 1, idx] = data[:, 3]

        M[idx, idx + 2] = data[:, 4]  # xz
        M[idx + 2, idx] = data[:, 4]

        M[idx + 1, idx + 2] = data[:, 5]  # yz
        M[idx + 2, idx + 1] = data[:, 5]

        return M