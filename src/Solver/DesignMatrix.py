import src.Preference.backups.Pre_Interface as Pre_Interface
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
import numpy as np
import h5py


class GravityDesignMat:
    """
    Gravity inversion in terms of spherical harmonic.
    """
    def __init__(self, arcNo: int, sat=Pre_Interface.sat, date_span=Pre_Interface.date_span):
        self._arcNo = arcNo
        self._date_span = date_span
        self._satName = None
        self._sat = [key for key, value in sat.items() if value]
        for sat in self._sat:
            if self._satName is None:
                self._satName = sat.name
            else:
                self._satName = self._satName + sat.name
        self._res = None
        self._accConfig = None
        self._solverConfig = None
        self._parameterConfig = None
        self._FMConfig = None
        self._ODEConfig = None
        self._AdjustConfig = None
        pass

    def configure(self, accConfig, solverConfig: SolverConfig, AdjustOrbitConfig: AdjustOrbitConfig,
                  ODEConfig:ODEConfig, FMConfig: ForceModelConfig, parameterConfig: ParameterConfig):
        self._accConfig = accConfig
        self._AdjustConfig = AdjustOrbitConfig
        self._solverConfig = solverConfig
        self._parameterConfig = parameterConfig
        self._FMConfig = FMConfig
        self._ODEConfig = ODEConfig
        self._res = self._solverConfig.DesignMatrixTemp
        return self

    def orbit_integration(self):
        adjustDM = AdjustDM()
        adjustDM.configure(SerDer=Orbit_var_2nd_diff, arcNo=self._arcNo, ParameterConfig=self._parameterConfig,
                           AdjustConfig=self._AdjustConfig,
                           FMConfig=self._FMConfig, SolverConfig=self._solverConfig, ODEConfig=self._ODEConfig)
        # '''define the directory of res output'''
        # res_dir = self._res.joinpath(self._date_span[0] + '_' + self._date_span[1])
        # if not res_dir.is_dir():
        #     res_dir.mkdir(exist_ok=True)
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

        res_dir = self._res + '/' + self._date_span[0] + '_' + self._date_span[1]
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
        res_dir = self._res + '/' + self._date_span[0] + '_' + self._date_span[1]
        res_filename = res_dir + '/Orbit_' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        DM_orbit = h5py.File(res_filename, 'r')

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

        '''make a record'''
        res_dir = self._res + '/' + self._date_span[0] + '_' + self._date_span[1]
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
        res_dir = self._res + '/' + self._date_span[0] + '_' + self._date_span[1]
        res_filename = res_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        orbit['r'] = h5['r'][()]
        orbit['v'] = h5['v'][()]
        orbit['t'] = h5['t'][()]
        h5.close()
        return orbit