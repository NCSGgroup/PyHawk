import src.Preference.Pre_Interface as Pre_Interface
import src.Preference.Pre_AdjustOrbit as Pre_AdjustOrbit
import src.Preference.Pre_NEQ as Pre_NEQ
from src.Preference.EnumType import Payload, SatID, SSTObserve
from src.Interface.ArcOperation import ArcData
import src.Preference.Pre_Solver as Pre_Solver
from src.Auxilary.GeoMathKit import GeoMathKit
import numpy as np
import pathlib
import h5py
from tqdm import tqdm
import time


class GravityNEQperArc:
    """
    Gravity inversion in terms of spherical harmonic.
    Retrieve the design matrix and observations, then transform the observation-equation into the normal equations.
    """
    def __init__(self, arcNo: int, sat=Pre_Interface.sat, date_span=Pre_Interface.date_span):
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
        self.__configSST = None
        self._sat = [key for key, value in sat.items() if value]

        for sat in self._sat:
            if self._satName is None:
                self._satName = sat.name
            else:
                self._satName = self._satName + sat.name

        res_dir = pathlib.Path(Pre_NEQ.PathOfFiles.NormalEqTemp).joinpath(date_span[0] + '_' + date_span[1])
        if not res_dir.is_dir():
            res_dir.mkdir(exist_ok=True)
        res_filename = res_dir.joinpath(str(arcNo) + '.hdf5')
        self.__saveData = h5py.File(res_filename, 'w')

        orbit_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp).joinpath(date_span[0] + '_' + date_span[1])
        orbit_filename = orbit_dir.joinpath(str(arcNo) + '_AB.hdf5')
        self.__orbitData = h5py.File(orbit_filename, 'r')
        pass

    def configure(self, **kwargs):
        self.__config['satA'] = kwargs.get("configA", Pre_NEQ.OrbitA)
        self.__config['satB'] = kwargs.get("configB", Pre_NEQ.OrbitB)
        self.__config['SST'] = kwargs.get("configSST", Pre_NEQ.SST)
        return self

    def get_neq(self):
        """
        three steps to acquire the normal equations. Assure the 'configure' is called before.
        :return:
        """
        res_dir = pathlib.Path(Pre_Solver.PathOfFiles.DesignMatrixTemp). \
            joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        res_filename = res_dir.joinpath('Orbit_' + str(self.__arcNo) + '_' + self._satName + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        self.__orbit_nominal = h5['r'][()]
        self.__t_nominal = h5['t'][()]
        self.__local_r = h5['local_r'][()]
        self.__global_r = h5['global_r'][()]
        h5.close()

        for i in range(3):
            if i == 0:
                self.get_neq_orbit(SatID.A)
            elif i == 1:
                self.get_neq_orbit(SatID.B)
            elif i == 2:
                self.get_neq_range_rate()
        return self

    def get_neq_orbit(self, sat: SatID):
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
        rd = ArcData(date_span=self.__date_span)

        if config.UseKinematicOrbit:
            orbit_nominal = rd.getData(arc=self.__arcNo, kind=Payload.KinematicOrbit, sat=sat)[:, 0:4]
        else:
            orbit_nominal = rd.getData(arc=self.__arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]
        orbit_obs = self.__orbitData['r'][:]
        t_obs = list(self.__orbitData['t'][:])
        '''for the orbit, only position (r) is needed for the inversion'''
        '''find the common'''

        int_t_obs = t_obs
        int_t_nominal = t_nominal.astype(np.int)
        new_index = [list(int_t_nominal).index(x) for x in int_t_obs]

        '''Reassignment'''
        t_nominal = t_nominal[new_index]
        orbit_nominal = orbit_nominal[new_index, :]
        local_r = local_r[new_index, :, :]
        global_r = global_r[new_index, :, :]

        if sat == SatID.A:
            orbit_obs = orbit_obs[:, 0:3, 0]
        else:
            orbit_obs = orbit_obs[:, 3:6, 0]
        obs = orbit_obs - orbit_nominal[:, 1:4]
        obs = obs.reshape((-1, 1))
        # TODO: Fix the kinematic orbit

        '''Expanding the design matrix'''
        n_local = np.shape(local_r)[-1]
        n_global = np.shape(global_r)[-1]
        local_r = local_r.reshape((-1, n_local))
        global_r = global_r.reshape((-1, n_global))

        N, l = GeoMathKit.keepGlobal(dm_global=global_r, dm_local=local_r, obs=obs)

        self.__saveData.create_dataset('Orbit_N_%1s' % sat.name, data=N)
        self.__saveData.create_dataset('Orbit_l_%1s' % sat.name, data=l)
        pass

    def get_neq_range_rate(self):
        """
        get Normal Equations for the range-rate only
        :return:
        """
        config = self.__config['SST']

        if not config.IsRequired:
            return None

        assert config.measurement == SSTObserve.RangeRate.value, 'Range-rate is not allowed by user setting'

        dm_dir = pathlib.Path(Pre_Solver.PathOfFiles.DesignMatrixTemp).joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        res_filename = dm_dir.joinpath(SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        t_dm = h5['t'][()]
        global_dm = h5['global'][()]
        local_dm = h5['local'][()]
        h5.close()

        dm_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.RangeRateTemp).joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        res_filename = dm_dir.joinpath(SSTObserve.RangeRate.name + '_' + str(self.__arcNo) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        t_obs = h5['post_t'][()]
        obs = h5['post_residual'][()]
        local_dm_empirical = h5['design_matrix'][()]
        h5.close()

        int_t_obs = t_obs.astype(np.int)
        int_t_dm = t_dm.astype(np.int)
        new_index = [list(int_t_dm).index(x) for x in int_t_obs]

        '''Reassignment'''
        t_dm = t_dm[new_index]
        global_dm = global_dm[new_index, :]
        local_dm = local_dm[new_index, :]

        '''Combination'''
        if config.CalibrateEmpiricalParameters:
            local_dm = np.concatenate((local_dm_empirical, local_dm), axis=1)

        N, l = GeoMathKit.keepGlobal(dm_global=global_dm, dm_local=local_dm, obs=obs)

        self.__saveData.create_dataset('%s_N' % SSTObserve.RangeRate.name, data=N)
        self.__saveData.create_dataset('%s_l' % SSTObserve.RangeRate.name, data=l)
        pass



