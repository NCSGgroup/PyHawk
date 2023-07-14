import h5py
from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType
import numpy as np
from src.ODE.Instance.GravimetryODE import GravimetryODE
from src.Frame.Frame import Frame, EOP
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_Solver import SolverConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_ODE import ODEConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_Frame import FrameConfig


class AdjustDM:
    def __init__(self, date_span, sat, kind=EnumType.Payload.GNV):
        self._date_span = date_span
        self._sat = sat
        self._kind = kind
        self._SerDer = None
        self._arcNo = None
        self._FMConfig = None
        self._AdjustConfig = None
        self._AdjustPathConfig = None
        self._ParameterConfig = None
        self._InterfaceConfig = None
        self._TransitionMatrixConfig = None
        self._AccelerometerConfig = None
        self._StokesCoefficientsConfig = None
        self._ODEConfig = None
        self._SolverConfig = None
        self._resPath = None
        self._initPath = None
        self._ArcData = None
        self._fr = None
        self._ODE = None
        self._satName = None
        for sat in self._sat:
            if self._satName is None:
                self._satName = sat
            else:
                self._satName = self._satName + sat
        pass

    def configure(self, SerDer, arcNo:int, ODEConfig:ODEConfig, AdjustConfig:AdjustOrbitConfig,
                  FMConfig: ForceModelConfig, ParameterConfig: ParameterConfig,
                  SolverConfig: SolverConfig, InterfaceConfig: InterfaceConfig,
                  FrameConfig: FrameConfig):
        self._SerDer = SerDer
        self._arcNo = arcNo
        self._FMConfig = FMConfig
        self._SolverConfig = SolverConfig
        self._AdjustConfig = AdjustConfig
        self._InterfaceConfig = InterfaceConfig
        self._resPath = self._SolverConfig.DesignMatrixTemp
        '''config adjust path'''
        self._AdjustPathConfig = self._AdjustConfig.PathOfFiles()
        self._AdjustPathConfig.__dict__.update(self._AdjustConfig.PathOfFilesConfig.copy())
        '''config ODE'''
        self._ODEConfig = ODEConfig
        '''config TransitionMatrix'''
        self._ParameterConfig = ParameterConfig
        self._TransitionMatrixConfig = self._ParameterConfig.TransitionMatrix()
        self._TransitionMatrixConfig.__dict__.update(self._ParameterConfig.TransitionMatrixConfig.copy())
        '''config AccelerometerConfig'''
        self._AccelerometerConfig = self._ParameterConfig.Accelerometer()
        self._AccelerometerConfig.__dict__.update(self._ParameterConfig.AccelerometerConfig.copy())
        '''config StokesCoefficientsConfig'''
        self._StokesCoefficientsConfig = self._ParameterConfig.StokesCoefficients()
        self._StokesCoefficientsConfig.__dict__.update(self._ParameterConfig.StokesCoefficientsConfig.copy())
        '''config ODE'''
        eop = EOP().configure(frameConfig=FrameConfig).load()
        self._fr = Frame(eop).configure(frameConfig=FrameConfig)
        self._ArcData = ArcData(interfaceConfig=self._InterfaceConfig)
        self._SerDer = SerDer(self._fr, self._ArcData).\
            configures(arc=self._arcNo, FMConfig=self._FMConfig, ParConfig=self._ParameterConfig, FrameConfig=FrameConfig).\
            setInitData(kind=self._kind)
        init_r, init_v = self.__getInit()
        startTime, _, _ = self._SerDer.getInitData()
        self._ODE = GravimetryODE().configure(ODEConfig=self._ODEConfig, ParameterConfig=self._ParameterConfig)\
            .setParNum().setInitial(startTime, init_r.copy(), init_v.copy()). \
            set2ndDerivative(self._SerDer)
        return self

    def calibrate(self):
        ode_res = self._ODE.propagate()
        self._save_StateVectors(ode_res)

    def _save_StateVectors(self, state):
        res_dir = self._resPath + '/' + self._date_span[0] + '_' + self._date_span[1]
        res_filename = res_dir + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        h5 = h5py.File(res_filename, 'w')
        h5.create_dataset('t', data=state[0])
        h5.create_dataset('r', data=state[1])
        h5.create_dataset('v', data=state[2])
        h5.close()
        pass

    def __getInit(self):
        self._initPath = self._AdjustPathConfig.OrbitAdjustResTemp
        self._initPath = self._initPath + '/' + self._date_span[0] + '_' + self._date_span[1]
        res_filename = self._initPath + '/' + str(self._arcNo) + '_' + self._satName + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        res = {}
        for key in h5:
            res[key] = h5[key][()]
        h5.close()

        a = res[EnumType.ParaType.TransitionMatrix.name]
        r_ini, v_ini = a[:, 0:3], a[:, 3:]
        return r_ini, v_ini