# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2024/10/14 15:20
# @File    : demo_1.py
# @Software: PyCharm
import os.path
import sys
sys.path.append("../")
import src.Preference.EnumType as EnumType
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Interface.KinematicOrbit import KinematicOrbitV3, KinematicOrbitV2, KinematicOrbitFO
from src.Interface.GapFix import GapFix
from src.Interface.ArcOperation import ArcSelect
from src.Interface.L1b import GRACE_NEW_RL02, GRACE_OLD_RL02, GRACE_RL03, GRACE_FO_RL04
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.Solver.AdjustOrbit import AdjustOrbit
from src.Preference.Pre_ODE import ODEConfig
import pathlib
from src.Preference.Pre_Accelerometer import AccelerometerConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Preference.Pre_Interface import InterfaceConfig
from src.Preference.Pre_Frame import FrameConfig
from src.Preference.Pre_CalibrateOrbit import CalibrateOrbitConfig
import multiprocessing
import json
from datetime import datetime


class demo_1:
    def __init__(self):
        self.__cur_path = os.path.abspath(__file__)
        self.__parent_path = os.path.abspath(os.path.dirname(self.__cur_path) + os.path.sep + "..")
        self.__acc_config_A = None
        self.__acc_config_B = None
        self.__force_model_config = None
        self.__parameter_config = None
        self.__frame_config = None
        self.__ode_config = None
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
        acc_A = json.load(open(os.path.join(self.__parent_path, 'setting/demo_1/AccelerometerConfig_A.json'), 'r'))
        self.__acc_config_A.__dict__ = acc_A

        self.__acc_config_B = AccelerometerConfig()
        acc_B = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_1/AccelerometerConfig_B.json'), 'r'))
        self.__acc_config_B.__dict__ = acc_B

        self.__force_model_config = ForceModelConfig()
        fmDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_1/ForceModelConfig.json'), 'r'))
        self.__force_model_config.__dict__ = fmDict

        self.__parameter_config = ParameterConfig()
        Parameter = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_1/ParameterConfig.json'), 'r'))
        self.__parameter_config.__dict__ = Parameter

        self.__adjust_config = AdjustOrbitConfig()
        adjustOrbitDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_1/AdjustOrbitConfig.json'), 'r'))
        self.__adjust_config.__dict__ = adjustOrbitDict

        self.__interface_config = InterfaceConfig()
        interfaceDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_1/InterfaceConfig.json'), 'r'))
        self.__interface_config.__dict__ = interfaceDict

        self.__frame_config = FrameConfig()
        frame = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_1/FrameConfig.json'), 'r'))
        self.__frame_config.__dict__ = frame

        self.__calibrateOrbit_config = CalibrateOrbitConfig()
        calibrateOrbit = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_1/CalibrateOrbitConfig.json'), 'r'))
        self.__calibrateOrbit_config.__dict__ = calibrateOrbit

        self.__ode_config = ODEConfig()
        odeDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_1/ODEConfig.json'), 'r'))
        self.__ode_config.__dict__ = odeDict
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

        isPreprocess = StepControl.isPreprocess
        isAdjustOrbit = StepControl.isAdjustOrbit
        AdjustOrbitProcess = ParallelControl.AdjustOrbitProcess

        if isPreprocess:
            self.makeLog(stepName='preprocess', process=1)
            self._preprocess(mission=MissionConfig.mission)

        with open(arc_path, "r", encoding='utf-8') as f:
            while True:
                data = f.readline().split(' ')
                if data[0] == 'Number':
                    self.__arcLen = int(data[3])
                    break
        self.__arclist = self.getArcList(path=arcft_path)
        if isAdjustOrbit:
            self.makeLog(stepName='adjustOrbit', process=AdjustOrbitProcess)
            pool = multiprocessing.Pool(processes=AdjustOrbitProcess)
            pool.map(self._adjustOrbit, self.__arclist)
            pool.close()
            pool.join()
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

    def _preprocess(self, mission):
        """step 1: Read raw data"""
        eop = EOP().configure(frameConfig=self.__frame_config).load()
        fr = Frame(eop).configure(frameConfig=self.__frame_config)
        s = None
        if mission == EnumType.Mission.GRACE_FO_RL04.name:
            s = GRACE_FO_RL04().configure(InterfaceConfig=self.__interface_config).setDate()
        elif mission == EnumType.Mission.GRACE_RL03.name:
            s = GRACE_RL03().configure(InterfaceConfig=self.__interface_config).setDate()
        s.read_double_sat()
        """step 2: Convert orbit from ITRS frame into GCRS frame"""
        kine = None
        if mission == EnumType.Mission.GRACE_FO_RL04.name:
            kine = KinematicOrbitFO(L1b=s).configure()
        if mission == EnumType.Mission.GRACE_RL03.name:
            kine = KinematicOrbitV2(L1b=s).configure()
        kine.read_double_sat()
        kine.GNVandKiOrbitItrs2Gcrs(fr)
        """step 3: Gap fix"""
        gf = GapFix(L1b=s).configure()
        gf.fix_all().makeReport()

        arc = ArcSelect(gf=gf).configure()
        arc.unpackArcs().makeReport().makeArcTF()

        pass

    def _adjustOrbit(self, i):
        """step 5: init AdjustOrbit"""
        AdjustOrbit(accConfig=[self.__acc_config_A, self.__acc_config_B], date_span=self.__date_span, sat=self.__sat).\
            configure(SerDer=Orbit_var_2nd_diff, arcNo=i, ODEConfig=self.__ode_config,
                      FMConfig=self.__force_model_config, AdjustOrbitConfig=self.__adjust_config,
                      ParameterConfig=self.__parameter_config, InterfaceConfig=self.__interface_config,
                      FrameConfig=self.__frame_config).calibrate(iteration=2)
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
    demo1 = demo_1()
    demo1.loadJson()
    demo1.run()
