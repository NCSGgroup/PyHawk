import h5py
from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType
import numpy as np
from src.ODE.Instance.GravimetryODE import GravimetryODE
from src.Interface.Accelerometer import Accelerometer, Accelerometer_V2, AccCaliPar, Accelerometer_V3, AccCaliPar_V3
from src.Frame.Frame import Frame, EOP
from src.Preference.Pre_Parameterization import ParameterConfig
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.Preference.Pre_Solver import SolverConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_ODE import ODEConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_Frame import FrameConfig
import matplotlib.pyplot as plt
from src.Auxilary.Outlier import Outlier
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# class AdjustOrbit:
#     def __init__(self, accConfig, date_span, sat, kind=EnumType.Payload.GNV):
#         # ======================= For demoTest===============================
#         self.fisrtR = None
#         self.secondR = None
#         self.thirdR = None
#         # ======================
#         self._fr = None
#         self._ode_res = None
#         self._state_path = None
#         self._state_dir = None
#         self._res_path = None
#         self._res_dir = None
#         self._res_h5 = None
#         self._ode = None
#         self._initTime = None
#         self._initR = None
#         self._endTime = None
#         self._endR = None
#         self._startTime = None
#         self._startPos = None
#         self._startVel = None
#         self._obsGNV = None
#         self._res = None
#         self._ap = {}
#         self._ac = {}
#         self._secDer = None
#         self._satName = None
#         self._date_span = date_span
#         self._arcNo = None
#         self._sat = [key for key, value in sat.items() if value]
#         self._kind = kind
#         self.accConfig = accConfig
#         self.ParameterConfig = None
#         self.TransitionMatrixConfig = None
#         self.AccelerometerConfig = None
#         self.StokesCoefficientsConfig = None
#         self._InterfaceConfig = None
#         self.FMConfig = None
#         self.ODEConfig = None
#         self.AdjustPath = None
#         self._commonTime = None
#
#     def configure(self, SerDer, arcNo, AdjustOrbitConfig:AdjustOrbitConfig,
#                   ODEConfig:ODEConfig, FMConfig: ForceModelConfig,
#                   ParameterConfig: ParameterConfig, InterfaceConfig: InterfaceConfig,
#                   FrameConfig:FrameConfig, **kwargs):
#         self._arcNo = arcNo
#         eop = EOP().configure(frameConfig=FrameConfig).load()
#         self._fr = Frame(eop).configure(frameConfig=FrameConfig)
#         '''config Interface Config'''
#         self._InterfaceConfig = InterfaceConfig
#         '''config Adjust path'''
#         self.AdjustPath = AdjustOrbitConfig.PathOfFiles()
#         self.AdjustPath.__dict__.update(AdjustOrbitConfig.PathOfFilesConfig.copy())
#         '''config ODE'''
#         self.ODEConfig = ODEConfig
#         '''config Force model'''
#         self.FMConfig = FMConfig
#         '''config TransitionMatrix'''
#         self.ParameterConfig = ParameterConfig
#         self.TransitionMatrixConfig = self.ParameterConfig.TransitionMatrix()
#         self.TransitionMatrixConfig.__dict__.update(self.ParameterConfig.TransitionMatrixConfig.copy())
#         '''config AccelerometerConfig'''
#         self.AccelerometerConfig = self.ParameterConfig.Accelerometer()
#         self.AccelerometerConfig.__dict__.update(self.ParameterConfig.AccelerometerConfig.copy())
#         '''config StokesCoefficientsConfig'''
#         self.StokesCoefficientsConfig = self.ParameterConfig.StokesCoefficients()
#         self.StokesCoefficientsConfig.__dict__.update(self.ParameterConfig.StokesCoefficientsConfig.copy())
#         '''define path'''
#         self._state_path = self.AdjustPath.StateVectorDataTemp
#         self._res_path = self.AdjustPath.OrbitAdjustResTemp
#         """define satName"""
#         for sat in self._sat:
#             if self._satName is None:
#                 self._satName = sat
#             else:
#                 self._satName = self._satName + sat
#         """Define the directory to deposit the state vectors"""
#         self._state_dir = self._state_path + '/' + self._date_span[0] + '_' + self._date_span[1]
#         if not os.path.exists(self._state_dir):
#             os.makedirs(self._state_dir, exist_ok=True)
#
#         '''define the directory of res output'''
#         self._res_dir = self._res_path + '/' + self._date_span[0] + '_' + self._date_span[1]
#         if not os.path.exists(self._res_dir):
#             os.makedirs(self._res_dir, exist_ok=True)
#
#         """step 1: combine the SCA and ACC data to obtain the non-conservative force as well as dadp arc by arc"""
#         rd = ArcData(interfaceConfig=self._InterfaceConfig)
#         for i in range(len(self._sat)):
#             ap = AccCaliPar(sat=self._sat[i], arcNo=self._arcNo, accConfig=self.accConfig[i])
#             ac = Accelerometer(rd=rd, ap=ap)
#             ac.get_and_save()
#             self._ac[self._sat[i]] = ac
#             self._ap[self._sat[i]] = ap
#
#         """step 3: instance second Derivative"""
#         self._secDer = SerDer(self._fr, rd).configures\
#             (arc=self._arcNo, FMConfig=self.FMConfig, ParConfig=self.ParameterConfig, FrameConfig=FrameConfig).\
#             setInitData(kind=self._kind)
#
#         """step 4: Propagator config"""
#         self._startTime, self._startPos, self._startVel = self._secDer.getInitData()
#         self._obsGNV = self._secDer.getInitState()
#         # self._commonTime = self._secDer.getCommonTime()
#         self._ode = GravimetryODE().configure(ODEConfig=self.ODEConfig, ParameterConfig=self.ParameterConfig).setParNum().\
#             setInitial(self._startTime, self._startPos.copy(), self._startVel.copy()).\
#             set2ndDerivative(self._secDer)
#         return self
#
#     def calibrate(self, iteration: int = 2):
#         # assert iteration >= 2
#         res_filename = self._res_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
#         self._res_h5 = h5py.File(res_filename, 'w')
#         ode = self._ode
#         '''step 1: orbit integration and save data'''
#         self._ode_res = ode.propagate()
#         self._save_StateVectors(self._ode_res)
#
#         # accA, accB = self._secDer.getacc()
#         # accA, accB = np.array(accA), np.array(accB)
#         # self.plot(accA[:, 0], accA[:, 1], accA[:, 2], 'A')
#         # self.plot(accB[:, 0], accB[:, 1], accB[:, 2], 'B')
#         filename = self._state_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
#         h5 = h5py.File(filename, 'r')
#         self._initTime = h5['t'][:]
#         self._initR = h5['r'][:]
#         h5.close()
#         '''step 2: orbit adjustment'''
#         orbit_init = np.hstack((self._startPos, self._startVel))
#         """TODO"""
#         acc_init = np.vstack((self._ap[self._sat[0]].getPar(), self._ap[self._sat[1]].getPar()))
#         '''step 3: start iteration'''
#         for i in range(iteration):
#             '''make the adjustment'''
#             self.__adjust(Time=self._ode_res[0], StateVectors=self._ode_res[1], isGNV=True)
#             orbit_ini, acc_ini = self.__updateIniValue(iniOrbit=orbit_init, iniAcc=acc_init)
#             r_ini, v_ini = orbit_ini[:, 0:3], orbit_ini[:, 3:]
#
#             '''step 4: update accelerometer'''
#             for j in range(len(self._sat)):
#                 self._ac[self._sat[j]].updatePar(paralist=acc_ini[j]).get_and_save()
#
#             '''step 5: update force model'''
#             self._secDer.reConfigure()
#
#             '''step 6: update initial state vector'''
#             ode.setInitial(self._startTime, r_ini, v_ini)
#
#             '''step 7: orbit integration once again'''
#             self._ode_res = ode.propagate()
#             self._save_StateVectors(self._ode_res)
#
#             if i == 0:
#                 self.fisrtR = self._ode_res[1]
#             elif i == 1:
#                 self.secondR = self._ode_res[1]
#             elif i == 2:
#                 self.thirdR = self._ode_res[1]
#         h5 = h5py.File(filename, 'r')
#         self._endTime = h5['t'][:]
#         self._endR = h5['r'][:]
#         h5.close()
#         self._res_h5.close()
#         return self
#
#     def __updateIniValue(self, iniOrbit: np.ndarray, iniAcc: np.ndarray):
#         res = self._res
#         h5 = self._res_h5
#
#         key = self.TransitionMatrixConfig
#         if key.isRequired:
#             iniOrbit += res[:, 0:key.Parameter_Number]
#             if self.ParameterConfig.TransitionMatrix.__name__ in h5:
#                 h5[self.ParameterConfig.TransitionMatrix.__name__][()] = iniOrbit
#             else:
#                 h5.create_dataset(self.ParameterConfig.TransitionMatrix.__name__, data=iniOrbit)
#
#         key = self.AccelerometerConfig
#         if key.isRequired:
#             left = self.TransitionMatrixConfig.Parameter_Number
#             right = left + key.Parameter_Number
#             iniAcc += res[:, left:right]
#             if self.ParameterConfig.Accelerometer.__name__ in h5:
#                 h5[self.ParameterConfig.Accelerometer.__name__][()] = iniAcc
#             else:
#                 h5.create_dataset(self.ParameterConfig.Accelerometer.__name__, data=iniAcc)
#         return iniOrbit, iniAcc
#
#     def __adjust(self,Time, StateVectors: np.ndarray, isGNV: bool, refOrbit: np.ndarray = None):
#         """
#         :param isGNV: is the reference orbit GNV ?
#         :param StateVectors: orbit integration output
#         :param refOrbit: reference orbit. It could be the GNV orbit or the other orbit like kinematic orbit transferred
#         from outside; The first column must be the time stamp, and the followings are x,y,z, and optionally vx,vy,vz.
#         :return:
#         """
#         StateTime = Time
#         xyz = StateVectors
#         # index = [list(StateTime).index(x) for x in self._commonTime]
#         # xyz = xyz[index, :, :]
#         obs = None
#         if isGNV:
#             obs = self._obsGNV
#         else:
#             obs = refOrbit
#         '''least-square solver'''
#         residual = obs - xyz[:, :, 0]
#
#         DesignMatrixA = xyz[:, 0:3, 1:].reshape((len(xyz) * 3, -1))
#         bA = residual[:, 0:3].flatten()
#         QA, RA = np.linalg.qr(DesignMatrixA)
#         xA = np.linalg.inv(RA).dot(QA.T).dot(bA)
#
#         # xA = np.linalg.lstsq(DesignMatrixA, bA, rcond=None)[0]
#
#         DesignMatrixB = xyz[:, 3:6, 1:].reshape((len(xyz) * 3, -1))
#         bB = residual[:, 3:6].flatten()
#         QB, RB = np.linalg.qr(DesignMatrixB)
#         xB = np.linalg.inv(RB).dot(QB.T).dot(bB)
#         # xB = np.linalg.lstsq(DesignMatrixB, bB, rcond=None)[0]
#
#         self._res = np.vstack((xA, xB))
#
#         return self
#
#     def _save_StateVectors(self, state):
#         res_dir = self._state_dir
#         res_filename = res_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
#         # print(res_filename)
#         h5 = h5py.File(res_filename, 'w')
#         h5.create_dataset('t', data=state[0])
#         h5.create_dataset('r', data=state[1])
#         h5.create_dataset('v', data=state[2])
#         h5.close()
#         pass
#
#     def draw(self):
#         return self._initTime, self._initR, self._endTime, self._endR
#
#     def getR(self):
#         return self.fisrtR, self.secondR, self.thirdR
#
#     def plot(self, x, y, z, sat):
#         t = np.arange(len(x))
#         plt.figure(figsize=(12, 6))
#         plt.subplot(311)
#         plt.title('{}-{}-{}-{}'.format('force', sat, '2021-01', 'arc1'), fontsize=24)
#         plt.scatter(t, x, marker='o')
#         plt.yticks(fontsize=18)
#         plt.ylabel(r'$X/m/s2$', fontsize=20)
#
#         plt.subplot(312)
#         plt.scatter(t, y, marker='o')
#         plt.yticks(fontsize=18)
#         plt.ylabel(r'$Y/m/s2$', fontsize=20)
#
#         plt.subplot(313)
#         plt.scatter(t, z, marker='o')
#         plt.xlabel('GPS Time', fontsize=20)
#         plt.ylabel(r'$Z/m/s2$', fontsize=20)
#         plt.yticks(fontsize=18)
#
#         # plt.grid(ls='--')
#         plt.show()

class AdjustOrbit:
    def __init__(self, accConfig, date_span, sat, kind=EnumType.Payload.GNV):
        # ======================= For demoTest===============================
        self.fisrtR = None
        self.secondR = None
        self.thirdR = None
        # ======================
        self._fr = None
        self._rd = None
        self._ode_res = None
        self._state_path = None
        self._state_dir = None
        self._res_path = None
        self._res_dir = None
        self._res_h5 = None
        self._ode = None
        self._initTime = None
        self._initR = None
        self._endTime = None
        self._endR = None
        self._startTime = None
        self._startPos = None
        self._startVel = None
        self._obsGNV = None
        self._res = None
        self._ap = {}
        self._ac = {}
        self._secDer = None
        self._satName = None
        self._date_span = date_span
        self._arcNo = None
        self._sat = [key for key, value in sat.items() if value]
        self._kind = kind
        self.accConfig = accConfig
        self.ParameterConfig = None
        self.TransitionMatrixConfig = None
        self.AccelerometerConfig = None
        self.StokesCoefficientsConfig = None
        self._InterfaceConfig = None
        self.FMConfig = None
        self.ODEConfig = None
        self.AccAdjustLength = None
        self.AdjustPath = None
        self._commonTime = None
        self._accParNum = 0
        self._accParNumBySat = {}

    def configure(self, SerDer, arcNo, AdjustOrbitConfig:AdjustOrbitConfig,
                  ODEConfig:ODEConfig, FMConfig: ForceModelConfig,
                  ParameterConfig: ParameterConfig, InterfaceConfig: InterfaceConfig,
                  FrameConfig:FrameConfig, **kwargs):
        self._arcNo = arcNo
        eop = EOP().configure(frameConfig=FrameConfig).load()
        self._fr = Frame(eop).configure(frameConfig=FrameConfig)
        '''config Interface'''
        self._InterfaceConfig = InterfaceConfig
        '''config Adjust path'''
        self.AdjustPath = AdjustOrbitConfig.PathOfFiles()
        self.AdjustPath.__dict__.update(AdjustOrbitConfig.PathOfFilesConfig.copy())
        self._AdjustConfig = AdjustOrbitConfig.Orbit()
        self._AdjustConfig.__dict__.update(AdjustOrbitConfig.OrbitConfig.copy())

        '''config ODE'''
        self.ODEConfig = ODEConfig
        '''config Force model'''
        self.FMConfig = FMConfig
        '''config TransitionMatrix'''
        self.ParameterConfig = ParameterConfig
        self.TransitionMatrixConfig = self.ParameterConfig.TransitionMatrix()
        self.TransitionMatrixConfig.__dict__.update(self.ParameterConfig.TransitionMatrixConfig.copy())
        '''config AccelerometerConfig'''
        self.AccelerometerConfig = self.ParameterConfig.Accelerometer()
        self.AccelerometerConfig.__dict__.update(self.ParameterConfig.AccelerometerConfig.copy())
        self.AccAdjustLength = self.AccelerometerConfig.AdjustLength
        self.X_AdjustLength = self.AccelerometerConfig.X_AdjustLength
        self.Y_AdjustLength = self.AccelerometerConfig.Y_AdjustLength
        self.Z_AdjustLength = self.AccelerometerConfig.Z_AdjustLength
        '''config StokesCoefficientsConfig'''
        self.StokesCoefficientsConfig = self.ParameterConfig.StokesCoefficients()
        self.StokesCoefficientsConfig.__dict__.update(self.ParameterConfig.StokesCoefficientsConfig.copy())
        '''define path'''
        self._state_path = self.AdjustPath.StateVectorDataTemp
        self._res_path = self.AdjustPath.OrbitAdjustResTemp
        """define satName"""
        for sat in self._sat:
            if self._satName is None:
                self._satName = sat
            else:
                self._satName = self._satName + sat
        """Define the directory to deposit the state vectors"""
        self._state_dir = self._state_path + '/' + self._date_span[0] + '_' + self._date_span[1]
        if not os.path.exists(self._state_dir):
            os.makedirs(self._state_dir, exist_ok=True)

        '''define the directory of res output'''
        self._res_dir = self._res_path + '/' + self._date_span[0] + '_' + self._date_span[1]
        if not os.path.exists(self._res_dir):
            os.makedirs(self._res_dir, exist_ok=True)

        """step 1: combine the SCA and ACC data to obtain the non-conservative force as well as dadp arc by arc"""
        self._rd = ArcData(interfaceConfig=self._InterfaceConfig)

        '''init acc'''

        for i in range(len(self._sat)):
            ap = AccCaliPar_V3(rd=self._rd, sat=self._sat[i], arcNo=self._arcNo,
                            accConfig=self.accConfig[i], adjustlength_x=self.X_AdjustLength,
                               adjustlength_y=self.Y_AdjustLength, adjustlength_z=self.Z_AdjustLength)
            # ap = AccCaliPar(rd=self._rd, sat=self._sat[i], arcNo=self._arcNo,
            #                 accConfig=self.accConfig[i], adjustlength=self.AccAdjustLength)
            # ac = Accelerometer_V2(ap=ap)
            ac = Accelerometer_V3(ap=ap)
            ac.get_and_save()

            self._ac[self._sat[i]] = ac
            self._ap[self._sat[i]] = ap
            self._accParNumBySat[self._sat[i]] = len(ap.getPar())

        if self._accParNumBySat:
            self._accParNum = max(self._accParNumBySat.values())

        """step 3: instance second Derivative"""
        self._secDer = SerDer(self._fr, self._rd).configures \
            (arc=self._arcNo, FMConfig=self.FMConfig, ParConfig=self.ParameterConfig, FrameConfig=FrameConfig). \
            setInitData(kind=EnumType.Payload.GNV)

        from src.Solver.OrbitObservations import orbit_payload
        self._kind=orbit_payload(AdjustOrbitConfig.OrbitConfig,self._kind)
        self._fit_observations={sat:self._rd.getData(arc=self._arcNo,kind=self._kind,sat=sat)
                                for sat in self._sat}
        times = self._secDer.getTimes()
        """step 4: Propagator config"""
        self._startTime, self._startPos, self._startVel = self._secDer.getInitData()
        self._obsGNV = self._secDer.getInitState()
        # self._commonTime = self._secDer.getCommonTime()
        self._ode = GravimetryODE().configure(ODEConfig=self.ODEConfig, ParameterConfig=self.ParameterConfig).\
            setDataTime(times).setParNum().setInitial(self._startTime, self._startPos.copy(), self._startVel.copy()).\
            set2ndDerivative(self._secDer)
        return self

    def calibrate(self, iteration: int = 2):
        # assert iteration >= 2
        ode = self._ode
        '''step 1: orbit integration and save data'''
        self._clear_acc_history()
        self._ode_res = ode.propagate()
        self._save_StateVectors(self._ode_res)
        filename = self._state_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        with h5py.File(filename, 'r') as h5:
            self._initTime = h5['t'][:]
            self._initR = h5['r'][:]

        '''step 2: orbit adjustment'''
        orbit_init = np.hstack((self._startPos, self._startVel))
        """TODO"""
        acc_init = np.zeros((len(self._sat), self._accParNum), dtype=float)
        for j, sat in enumerate(self._sat):
            par = self._ap[sat].getPar()
            acc_init[j, :len(par)] = par
        '''step 3: start iteration'''
        for i in range(iteration):
            '''make the adjustment'''
            if self._kind == EnumType.Payload.GNV and not getattr(self._AdjustConfig,'SamplingSeconds',0.) and not getattr(self._AdjustConfig,'UseEpochCovariance',False):
                self.__adjust(Time=self._ode_res[0],StateVectors=self._ode_res[1],isGNV=True)
            else:
                self.__adjust_positions(self._ode_res[0],self._ode_res[1])
            orbit_ini, acc_ini = self.__updateIniValue(iniOrbit=orbit_init, iniAcc=acc_init)
            r_ini, v_ini = orbit_ini[:, 0:3], orbit_ini[:, 3:]

            '''step 4: update accelerometer'''
            for j in range(len(self._sat)):
                sat = self._sat[j]
                par_num = self._accParNumBySat[sat]
                self._ac[sat].updatePar(paralist=acc_ini[j, :par_num]).get_and_save()

            '''step 5: update force model'''
            self._secDer.reConfigure()

            '''step 6: update initial state vector'''
            ode.setInitial(self._startTime, r_ini, v_ini)

            '''step 7: orbit integration once again'''
            self._clear_acc_history()
            self._ode_res = ode.propagate()
            # if i == iteration - 1:
            #     orbit_res = self._ode_res[1][:, :, 0] - self._obsGNV
            #     index = Outlier(rms_times=self._AdjustConfig.OutlierTimes, upper_RMS=self._AdjustConfig.OutlierLimit).remove_V2(
            #         self._ode_res[0], orbit_res[:, 0])
            #     self._ode_res[0] = self._ode_res[0][index]
            #     self._ode_res[1] = self._ode_res[1][index, :, :]
            #     self._ode_res[2] = self._ode_res[2][index, :, :]
            self._save_StateVectors(self._ode_res)

        with h5py.File(filename, 'r') as h5:
            self._endTime = h5['t'][:]
            self._endR = h5['r'][:]

        return self

    def __updateIniValue(self, iniOrbit: np.ndarray, iniAcc: np.ndarray):
        res = self._res

        res_filename = self._res_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        with h5py.File(res_filename, 'w') as h5:
            h5.attrs['orbit_observation_source']=self._kind.name
            h5.attrs['initial_seed_source']='GNV position/velocity; observations fitted separately'
            key = self.TransitionMatrixConfig
            if key.isRequired:
                iniOrbit += res[:, 0:key.Parameter_Number]
                if self.ParameterConfig.TransitionMatrix.__name__ in h5:
                    h5[self.ParameterConfig.TransitionMatrix.__name__][()] = iniOrbit
                else:
                    h5.create_dataset(self.ParameterConfig.TransitionMatrix.__name__, data=iniOrbit)

            key = self.AccelerometerConfig
            if key.isRequired:
                left = self.TransitionMatrixConfig.Parameter_Number
                iniAcc += res[:, left:]
                if self.ParameterConfig.Accelerometer.__name__ in h5:
                    h5[self.ParameterConfig.Accelerometer.__name__][()] = iniAcc
                else:
                    h5.create_dataset(self.ParameterConfig.Accelerometer.__name__, data=iniAcc)
        return iniOrbit, iniAcc

    def __adjust_positions(self,time,state):
        from src.Solver.OrbitObservations import fit_position_increment
        epoch_weighting=bool(getattr(self._AdjustConfig,'UseEpochCovariance',False))
        if epoch_weighting and self._kind != EnumType.Payload.KinematicOrbit:
            raise ValueError('UseEpochCovariance requires KIN observations')
        increments=[]
        for i,sat in enumerate(self._sat):
            increments.append(fit_position_increment(self._fit_observations[sat],time,
                state[:,i*3:i*3+3,:],epoch_weighting,
                float(getattr(self._AdjustConfig,'SamplingSeconds',0.)),
                getattr(self._AdjustConfig,'SamplingOrigin','GPS')))
        self._res=np.vstack(increments)

    def __adjust(self,Time, StateVectors: np.ndarray, isGNV: bool, refOrbit: np.ndarray = None):
        """
        :param isGNV: is the reference orbit GNV ?
        :param StateVectors: orbit integration output
        :param refOrbit: reference orbit. It could be the GNV orbit or the other orbit like kinematic orbit transferred
        from outside; The first column must be the time stamp, and the followings are x,y,z, and optionally vx,vy,vz.
        :return:
        """
        StateTime = Time
        xyz = StateVectors
        # index = [list(StateTime).index(x) for x in self._commonTime]
        # xyz = xyz[index, :, :]
        obs = None
        if isGNV:
            obs = self._obsGNV
        else:
            obs = refOrbit
        '''least-square solver'''
        residual = obs - xyz[:, :, 0]

        DesignMatrixA = xyz[:, 0:3, 1:].reshape((len(xyz) * 3, -1))
        bA = residual[:, 0:3].flatten()
        # QA, RA = np.linalg.qr(DesignMatrixA)
        # xA = np.linalg.solve(RA, QA.T @ bA)

        xA = np.linalg.lstsq(DesignMatrixA, bA, rcond=None)[0]

        DesignMatrixB = xyz[:, 3:6, 1:].reshape((len(xyz) * 3, -1))
        bB = residual[:, 3:6].flatten()
        # QB, RB = np.linalg.qr(DesignMatrixB)
        # xB = np.linalg.solve(RB, QB.T @ bB)

        xB = np.linalg.lstsq(DesignMatrixB, bB, rcond=None)[0]

        self._res = np.vstack((xA, xB))

        return self

    def _clear_acc_history(self):
        if hasattr(self._secDer, 'clearAccHistory'):
            self._secDer.clearAccHistory()

    def _save_StateVectors(self, state):
        res_dir = self._state_dir
        os.makedirs(res_dir, exist_ok=True)

        res_filename = res_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        state_time = np.asarray(state[0], dtype=float).reshape(-1)
        # print(res_filename)
        with h5py.File(res_filename, 'w') as h5:
            h5.create_dataset('t', data=state[0])
            h5.create_dataset('r', data=state[1])
            h5.create_dataset('v', data=state[2])
            self.__save_acceleration_history(h5, state_time)
        pass

    def __save_acceleration_history(self, h5, state_time):
        if not hasattr(self._secDer, 'getAccWithTime'):
            return

        acc_time, accA, accB = self._secDer.getAccWithTime()
        accA = self.__align_acceleration_history(state_time, acc_time, accA)
        accB = self.__align_acceleration_history(state_time, acc_time, accB)
        if accA is None or accB is None:
            return

        h5.create_dataset('acc_time', data=state_time)
        h5.create_dataset('accA', data=accA)
        h5.create_dataset('accB', data=accB)
        h5.attrs['acc_source'] = 'total force-model acceleration from secDerivative'
        h5.attrs['acc_frame'] = 'GCRS/inertial frame used by orbit integration'
        h5.attrs['acc_alignment'] = 'deduplicated derivative evaluations interpolated to saved state-vector epochs'

    @staticmethod
    def __align_acceleration_history(state_time, acc_time, acc):
        state_time = np.asarray(state_time, dtype=float).reshape(-1)
        acc_time = np.asarray(acc_time, dtype=float).reshape(-1)
        acc = np.asarray(acc, dtype=float)

        if len(state_time) == 0 or len(acc_time) == 0 or acc.size == 0:
            return None

        acc = np.squeeze(acc)
        if acc.ndim == 1:
            if acc.size % 3 != 0:
                return None
            acc = acc.reshape((-1, 3))
        if acc.ndim != 2 or acc.shape[1] != 3:
            return None

        count = min(len(acc_time), len(acc))
        acc_time = acc_time[:count]
        acc = acc[:count]

        finite = np.isfinite(acc_time) & np.all(np.isfinite(acc), axis=1)
        acc_time = acc_time[finite]
        acc = acc[finite]
        if len(acc_time) == 0:
            return None

        order = np.argsort(acc_time)
        acc_time = acc_time[order]
        acc = acc[order]

        rounded_time = np.round(acc_time, 9)
        _, last_reversed = np.unique(rounded_time[::-1], return_index=True)
        keep = len(rounded_time) - 1 - last_reversed
        keep = np.sort(keep)
        acc_time = acc_time[keep]
        acc = acc[keep]

        if len(acc_time) == 1:
            return np.repeat(acc[:1], len(state_time), axis=0)

        return np.column_stack([
            np.interp(state_time, acc_time, acc[:, i])
            for i in range(3)
        ])

    def _setInitState(self):

        pass

    def draw(self):
        return self._initTime, self._initR, self._endTime, self._endR

    def getR(self):
        return self.fisrtR, self.secondR, self.thirdR

    def plot(self, x, y, z, sat):
        t = np.arange(len(x))
        plt.figure(figsize=(12, 6))
        plt.subplot(311)
        plt.title('{}-{}-{}-{}'.format('force', sat, '2021-01', 'arc1'), fontsize=24)
        plt.scatter(t, x, marker='o')
        plt.yticks(fontsize=18)
        plt.ylabel(r'$X/m/s2$', fontsize=20)

        plt.subplot(312)
        plt.scatter(t, y, marker='o')
        plt.yticks(fontsize=18)
        plt.ylabel(r'$Y/m/s2$', fontsize=20)

        plt.subplot(313)
        plt.scatter(t, z, marker='o')
        plt.xlabel('GPS Time', fontsize=20)
        plt.ylabel(r'$Z/m/s2$', fontsize=20)
        plt.yticks(fontsize=18)

        # plt.grid(ls='--')
        plt.show()
