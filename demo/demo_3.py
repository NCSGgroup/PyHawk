# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2024/10/14 15:22
# @File    : demo_3.py
# @Software: PyCharm
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
from src.Auxilary.SortCS import SortCS
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Interface.LoadSH import LoadGif48
from src.Preference.Pre_Accelerometer import AccelerometerConfig
from src.Preference.Pre_Solver import SolverConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_ODE import ODEConfig
from src.Preference.Pre_NEQ import NEQConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_Frame import FrameConfig
from src.Preference.Pre_CalibrateOrbit import CalibrateOrbitConfig
from src.Solver.AdjustSST import RangeRate
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Interface.KinematicOrbit import KinematicOrbitV3, KinematicOrbitV2, KinematicOrbitFO
from src.Interface.GapFix import GapFix
from src.Interface.ArcOperation import ArcSelect
from src.Interface.L1b import GRACE_NEW_RL02, GRACE_OLD_RL02, GRACE_RL03, GRACE_FO_RL04
from src.Auxilary.Format import FormatWrite
import multiprocessing
import json
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
        self.__frame_config = None
        self.__solver_config = None
        self.__ode_config = None
        self.__neq_config = None
        self.__adjust_config = None
        self.__interface_config = None
        self.__calibrateOrbit_config = None
        self.__arcLen = None
        self.__date_span = None
        self.__sat = None
        self.__log_path = None
        self.__arclist = []
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

        self.__frame_config = FrameConfig()
        frame = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/FrameConfig.json'), 'r'))
        self.__frame_config.__dict__ = frame

        self.__calibrateOrbit_config = CalibrateOrbitConfig()
        calibrateOrbit = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/CalibrateOrbitConfig.json'), 'r'))
        self.__calibrateOrbit_config.__dict__ = calibrateOrbit
        return self

    def run(self):
        '''get arc number'''
        interface_config = self.__interface_config
        self.InterfacePathConfig = InterfaceConfig.PathOfFiles(interface_config)
        self.InterfacePathConfig.__dict__.update(interface_config.PathOfFilesConfig.copy())
        self.__log_path = self.InterfacePathConfig.log
        self.__date_span = interface_config.date_span
        self.__sat = interface_config.sat
        arc_path = pathlib.Path().joinpath(self.InterfacePathConfig.report_arc, self.__date_span[0]+'_'+self.__date_span[1]+'.txt')
        arcft_path = pathlib.Path().joinpath(self.InterfacePathConfig.report_arcft, self.__date_span[0]+'_'+self.__date_span[1]+'.txt')
        self.__arcLen = None

        '''get step and process control'''
        calibrateOrbit_config = self.__calibrateOrbit_config
        MissionConfig = CalibrateOrbitConfig.Mission()
        MissionConfig.__dict__.update(calibrateOrbit_config.MissionConfig.copy())
        StepControl = CalibrateOrbitConfig.StepControl()
        StepControl.__dict__.update(calibrateOrbit_config.StepControlConfig.copy())
        ParallelControl = CalibrateOrbitConfig.ParallelControl()
        ParallelControl.__dict__.update(calibrateOrbit_config.ParallelControlConfig.copy())

        isNEQ = StepControl.isGetNEQ
        isSH = StepControl.isGetSH

        GetNEQProcess = ParallelControl.GetNEQProcess

        with open(arc_path, "r", encoding='utf-8') as f:
            while True:
                data = f.readline().split(' ')
                if data[0] == 'Number':
                    self.__arcLen = int(data[3])
                    break
        self.__arclist = self.getArcList(path=arcft_path)

        if isNEQ:
            self.makeLog(stepName='NEQ', process=GetNEQProcess)
            pool = multiprocessing.Pool(processes=GetNEQProcess)
            pool.map(self._NEQ, self.__arclist)
            # pool.map(self._NEQ, range(0, 1))
            pool.close()
            pool.join()
        if isSH:
            self.makeLog(stepName='SH', process=1)
            self.__calculate_cs()
        # self.getSpaceKBRR()
        pass

    def makeLog(self, stepName, process):
        path = self.__log_path + '/' + stepName
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path = path + '/' + self.__date_span[0] + '_' + self.__date_span[1] + '.txt'
        with open(path, 'w') as f:
            f.write('This ia a summary of log information.\n')
            f.write('Created by Wu Yi(wu_yi@hust.edu.cn)\n')
            f.write('begain time: %s \n\n' % datetime.now())
            f.write('process: %s \n\n' % process)
        pass

    def _NEQ(self, q):
        '''setp 8: get NEQ'''
        neq = GravityNEQperArc(arcNo=q, date_span=self.__date_span, sat=self.__sat)
        neq.configure(solverConfig=self.__solver_config, NEQConfig=self.__neq_config,
                      AdjustOrbitConfig=self.__adjust_config, InterfaceConfig=self.__interface_config).get_neq()
        pass

    def __calculate_cs(self):
        '''orbit and kbrr factor'''
        OrbitKinFactor = int(self.__solver_config.OrbitKinFactor)
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

        for i in tqdm(self.__arclist, desc='Solve normal equations: '):
            res_filename = res_dir + '/' + str(i) + '.hdf5'
            print(res_filename)
            h5 = h5py.File(res_filename, 'r')
            NMatrix = NMatrix + (h5['Orbit_N_A'][:] + h5['Orbit_N_B'][:]) + h5['RangeRate_N'][:] * OrbitKinFactor
            LMatrix = LMatrix + (h5['Orbit_l_A'][:] + h5['Orbit_l_B'][:]) + h5['RangeRate_l'][:].reshape(
                (parameter_number, 1)) * OrbitKinFactor
            h5.close()

        cs = np.linalg.lstsq(NMatrix, LMatrix[:, 0], rcond=None)[0]

        sort = SortCS(method=stokes_coefficients_config.SortMethod,
                      degree_max=stokes_coefficients_config.MaxDegree,
                      degree_min=stokes_coefficients_config.MinDegree)
        c, s = sort.invert(cs)

        static_c, static_s = LoadGif48().load(force_model_path_config.Gif48).getCS(stokes_coefficients_config.MaxDegree)

        C, S = c + static_c, s + static_s

        sigmaC , sigmaS = np.zeros(len(C)), np.zeros(len(S))

        format = FormatWrite().configure(filedir=neq_path_config.ResultCS,
                                         data=date_span, degree=stokes_coefficients_config.MaxDegree,
                                         c=C, s=S, sigmaC=sigmaC, sigmaS=sigmaS)
        print(len(C))
        format.unfiltered()

        pass

    def getArcList(self, path):
        arclist = []
        with open(path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(' ')
                if data[0] == 'Arc':
                    if data[3] == 'True':
                        arclist.append(int(data[2]))
        return arclist


if __name__ == '__main__':
    pass