import os.path
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
import multiprocessing
from datetime import datetime


class CalibrateOrbit:
    def __init__(self):
        self.__cur_path = os.path.abspath(__file__)
        self.__parent_path = os.path.abspath(os.path.dirname(self.__cur_path) + os.path.sep + ".." + os.path.sep + "..")
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
        self.__arcLen = None
        self.__date_span = None
        self.__sat = None
        pass

    def loadJson(self):
        self.__acc_config_A = AccelerometerConfig()
        acc_A = json.load(open(os.path.join(self.__parent_path, 'setting/Calibrate/AccelerometerConfig_A.json'), 'r'))
        self.__acc_config_A.__dict__ = acc_A

        self.__acc_config_B = AccelerometerConfig()
        acc_B = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/AccelerometerConfig_B.json'), 'r'))
        self.__acc_config_B.__dict__ = acc_B

        self.__force_model_config = ForceModelConfig()
        fmDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/ForceModelConfig.json'), 'r'))
        self.__force_model_config.__dict__ = fmDict

        self.__parameter_config = ParameterConfig()
        Parameter = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/ParameterConfig.json'), 'r'))
        self.__parameter_config.__dict__ = Parameter

        self.__solver_config = SolverConfig()
        solverDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/SolverConfig.json'), 'r'))
        self.__solver_config.__dict__ = solverDict

        self.__ode_config = ODEConfig()
        odeDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/ODEConfig.json'), 'r'))
        self.__ode_config.__dict__ = odeDict

        self.__neq_config = NEQConfig()
        neqDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/NEQConfig.json'), 'r'))
        self.__neq_config.__dict__ = neqDict

        self.__adjust_config = AdjustOrbitConfig()
        adjustOrbitDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/AdjustOrbitConfig.json'), 'r'))
        self.__adjust_config.__dict__ = adjustOrbitDict

        self.__interface_config = InterfaceConfig()
        interfaceDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/InterfaceConfig.json'), 'r'))
        self.__interface_config.__dict__ = interfaceDict

        self.__design_parameter_config = ParameterConfig()
        designParameter = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/DesignParameterConfig.json'), 'r'))
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

        self._parallel()
        self.__calculate_cs()
        pass
 
    def _parallel(self):
        interface_config = self.__interface_config
        self.__date_span = interface_config.date_span
        self.__sat = interface_config.sat
        date_list = GeoMathKit.getEveryDay(*self.__date_span)
        days = datetime.strptime(date_list[-1], '%Y-%m-%d') - datetime.strptime(date_list[0], '%Y-%m-%d')
        self.__arcLen = days.days * (24 / interface_config.arc_length)
        step = int(self.__arcLen / 6)
        p1 = multiprocessing.Process(target=self._unit, args=(0, step))
        p2 = multiprocessing.Process(target=self._unit, args=(step, step*2))
        p3 = multiprocessing.Process(target=self._unit, args=(step*2, step*3))
        p4 = multiprocessing.Process(target=self._unit, args=(step*3, step*4))
        p5 = multiprocessing.Process(target=self._unit, args=(step*4, step*5))
        p6 = multiprocessing.Process(target=self._unit, args=(step*5, step*6))
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        pass

    def _unit(self, start, end):
        for i in range(start, end):
            """step 5: init AdjustOrbit"""
            AdjustOrbit(accConfig=[self.__acc_config_A, self.__acc_config_B], date_span=self.__date_span, sat=self.__sat).\
                configure(SerDer=Orbit_var_2nd_diff, arcNo=i, ODEConfig=self.__ode_config,
                          FMConfig=self.__force_model_config, AdjustOrbitConfig=self.__adjust_config,
                          ParameterConfig=self.__parameter_config).calibrate(iteration=2)

            RR = RangeRate(arcNo=i, date_span=self.__date_span)
            RR.configure(AdjustOrbitConfig=self.__adjust_config)
            RR.calibrate()

            '''step 6: get DesignMatrix'''
            dm = GravityDesignMat(arcNo=i, date_span=self.__date_span, sat=self.__sat).\
                configure(accConfig=[self.__acc_config_A, self.__acc_config_B],
                          ODEConfig=self.__ode_config, AdjustOrbitConfig=self.__adjust_config,
                          solverConfig=self.__solver_config, FMConfig=self.__force_model_config,
                          parameterConfig=self.__design_parameter_config)
            dm.orbit_integration()
            dm.get_orbit_design_matrix()
            dm.get_sst_design_matrix()

            '''setp 8: get NEQ'''
            neq = GravityNEQperArc(arcNo=i, date_span=self.__date_span, sat=self.__sat)
            neq.configure(solverConfig=self.__solver_config, NEQConfig=self.__neq_config,
                          AdjustOrbitConfig=self.__adjust_config).get_neq()
        pass

    def __calculate_cs(self):
        neq_path_config = self.__neq_config.PathOfFiles()
        neq_path_config.__dict__.update(self.__neq_config.PathOfFilesConfig.copy())
        interface_config = self.__interface_config
        stokes_coefficients_config = self.__design_parameter_config.StokesCoefficients()
        stokes_coefficients_config.__dict__.update(self.__design_parameter_config.StokesCoefficientsConfig.copy())
        '''config force model path'''
        force_model_path_config = self.__force_model_config.PathOfFiles()
        force_model_path_config.__dict__.update(self.__force_model_config.PathOfFilesConfig.copy())

        res_dir = neq_path_config.NormalEqTemp + '/' +\
                  interface_config.date_span[0] + '_' + interface_config.date_span[1]
        result_name = neq_path_config.ResultCS + '/' + \
                      interface_config.date_span[0] + '_' + interface_config.date_span[1] + '.hdf5'

        parameter_number = stokes_coefficients_config.Parameter_Number
        NMatrix = np.zeros((parameter_number, parameter_number))
        LMatrix = np.zeros((parameter_number, 1))

        date_span = interface_config.date_span
        date_list = GeoMathKit.getEveryDay(*date_span)
        days = datetime.strptime(date_list[-1], '%Y-%m-%d') - datetime.strptime(date_list[0], '%Y-%m-%d')
        self.__arcLen = days.days * (24 / interface_config.arc_length)
        for i in tqdm(range(0, int(self.__arcLen)), desc='Solve normal equations: '):
            res_filename = res_dir + '/' + str(i) + '.hdf5'
            h5 = h5py.File(res_filename, 'r')
            NMatrix = NMatrix + h5['Orbit_N_A'][:] + h5['Orbit_N_B'][:] + h5['RangeRate_N'][:] * 2.5E11
            LMatrix = LMatrix + h5['Orbit_l_A'][:] + h5['Orbit_l_B'][:] + h5['RangeRate_l'][:].reshape(
                (parameter_number, 1)) * 2.5E11

        cs = np.linalg.lstsq(NMatrix, LMatrix[:, 0], rcond=None)[0]

        sort = SortCS(method=stokes_coefficients_config.SortMethod,
                      degree_max=stokes_coefficients_config.MaxDegree,
                      degree_min=stokes_coefficients_config.MinDegree)
        c, s = sort.invert(cs)

        static_c, static_s = LoadGif48().load(force_model_path_config.Gif48).getCS(stokes_coefficients_config.MaxDegree)

        C, S = c + static_c, s + static_s

        C, S = GeoMathKit.CS_1dTo2d(C), GeoMathKit.CS_1dTo2d(S)

        h5 = h5py.File(result_name, 'w')
        h5.create_dataset('c', data=C)
        h5.create_dataset('s', data=S)
        h5.close()
        pass

    def drawSH(self):

        pass