import sys
sys.path.append("../")

from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType
import numpy as np
from tqdm import tqdm
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.Solver.AdjustOrbit import AdjustOrbit
import matplotlib.pyplot as plt
import pathlib
import h5py
import multiprocessing
from src.Solver.DesignMatrix import GravityDesignMat
from src.Solver.NEQ import GravityNEQperArc
import src.Preference.backups.Pre_Interface as Pre_Interface
from src.Auxilary.SortCS import SortCS
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Interface.LoadSH import LoadGif48
from src.Preference.Pre_Accelerometer import AccelerometerConfig
from src.Preference.Pre_Solver import SolverConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Frame.EOP import EOP
from src.Interface.KinematicOrbit import KinematicOrbit
from src.Interface.GapFix import GapFix
from src.Interface.ArcOperation import ArcSelect
from src.Solver.AdjustSST import RangeRate
from src.Frame.Frame import Frame
from src.Interface.L1b import GRACE_NEW_RL02, GRACE_OLD_RL02
import json
from src.Preference.Pre_ODE import ODEConfig
from src.Preference.Pre_NEQ import NEQConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Preference.Pre_Interface import InterfaceConfig


class CalibrateOrbit:
    def __init__(self):
        self.__acc_config_A = None
        self.__acc_config_B = None
        self.__force_model_config = None
        self.__parameter_config = None
        self.__design_parameter_config = None
        self.__solver_config = None
        self.__ode_config = None
        self.__neq_config = None
        self.__adjust_config = None
        self.__interface_config = None
        pass

    def loadJson(self):
        self.__acc_config_A = AccelerometerConfig()
        acc_A = json.load(open('../setting/Calibrate/AccelerometerConfig_A.json', 'r'))
        self.__acc_config_A.__dict__ = acc_A

        self.__acc_config_B = AccelerometerConfig()
        acc_B = json.load(open('../setting/Calibrate/AccelerometerConfig_B.json', 'r'))
        self.__acc_config_B.__dict__ = acc_B

        self.__force_model_config = ForceModelConfig()
        fmDict = json.load(open('../setting/Calibrate/ForceModelConfig.json', 'r'))
        self.__force_model_config.__dict__ = fmDict

        self.__parameter_config = ParameterConfig()
        Parameter = json.load(open('../setting/Calibrate/ParameterConfig.json', 'r'))
        self.__parameter_config.__dict__ = Parameter

        self.__solver_config = SolverConfig()
        solverDict = json.load(open('../setting/Calibrate/SolverConfig.json', 'r'))
        self.__solver_config.__dict__ = solverDict

        self.__ode_config = ODEConfig()
        odeDict = json.load(open('../setting/Calibrate/ODEConfig.json', 'r'))
        self.__ode_config.__dict__ = odeDict

        self.__neq_config = NEQConfig()
        neqDict = json.load(open('../setting/Calibrate/NEQConfig.json', 'r'))
        self.__neq_config.__dict__ = neqDict

        self.__adjust_config = AdjustOrbitConfig()
        adjustOrbitDict = json.load(open('../setting/Calibrate/AdjustOrbitConfig.json', 'r'))
        self.__adjust_config.__dict__ = adjustOrbitDict

        self.__interface_config = InterfaceConfig()
        interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
        self.__interface_config.__dict__ = interfaceDict

        self.__design_parameter_config = ParameterConfig()
        designParameter = json.load(open('../setting/Calibrate/DesignParameterConfig.json', 'r'))
        self.__design_parameter_config.__dict__ = designParameter
        return self

    def run(self):
        """step 1: Read raw data"""
        fr = Frame(EOP().load())
        s = GRACE_OLD_RL02().configure(InterfaceConfig=self.__interface_config).setDate()
        s.read_double_sat()
        """step 2: Convert orbit from ITRS frame into GCRS frame"""
        kine = KinematicOrbit(L1b=s).configure()
        # kine.read_double_sat()
        kine.GNVandKiOrbitItrs2Gcrs(fr)
        """step 3: Gap fix"""
        gf = GapFix(L1b=s).configure()
        gf.fix_all().makeReport()
        arc = ArcSelect(gf=gf).configure()
        arc.unpackArcs().makeReport()
        """step 5: init AdjustOrbit"""
        date_span = ['2008-01-01', '2008-01-31']
        AdjustOrbit(accConfig=[self.__acc_config_A, self.__acc_config_B], date_span=date_span). \
            configure(SerDer=Orbit_var_2nd_diff, arcNo=0, ODEConfig=self.__ode_config,
                      FMConfig=self.__force_model_config, AdjustOrbitConfig=self.__adjust_config,
                      ParameterConfig=self.__parameter_config).calibrate(iteration=2)

        RR = RangeRate(0)
        RR.configure(AdjustOrbitConfig=self.__adjust_config)
        RR.calibrate()
        '''step 6: get DesignMatrix'''
        dm = GravityDesignMat(0).configure(accConfig=[self.__acc_config_A, self.__acc_config_B],
                                           ODEConfig=self.__ode_config, AdjustOrbitConfig=self.__adjust_config,
                                           solverConfig=self.__solver_config, FMConfig=self.__force_model_config,
                                           parameterConfig=self.__design_parameter_config)
        dm.orbit_integration()
        dm.get_orbit_design_matrix()
        dm.get_sst_design_matrix()
        '''setp 8: get NEQ'''
        neq = GravityNEQperArc(0)
        neq.configure(solverConfig=self.__solver_config, NEQConfig=self.__neq_config,
                      AdjustOrbitConfig=self.__adjust_config).get_neq()

        pass
