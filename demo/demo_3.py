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


class demo_3:
    def __init__(self):
        self.__cur_path = os.path.abspath(__file__)
        self.__parent_path = os.path.abspath(os.path.dirname(self.__cur_path) + os.path.sep + "..")
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
        acc_A = json.load(open(os.path.join(self.__parent_path, 'setting/demo_3/AccelerometerConfig_A.json'), 'r'))
        self.__acc_config_A.__dict__ = acc_A

        self.__acc_config_B = AccelerometerConfig()
        acc_B = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/AccelerometerConfig_B.json'), 'r'))
        self.__acc_config_B.__dict__ = acc_B

        self.__force_model_config = ForceModelConfig()
        fmDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/ForceModelConfig.json'), 'r'))
        self.__force_model_config.__dict__ = fmDict

        self.__parameter_config = ParameterConfig()
        Parameter = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/ParameterConfig.json'), 'r'))
        self.__parameter_config.__dict__ = Parameter

        self.__solver_config = SolverConfig()
        solverDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/SolverConfig.json'), 'r'))
        self.__solver_config.__dict__ = solverDict

        self.__ode_config = ODEConfig()
        odeDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/ODEConfig.json'), 'r'))
        self.__ode_config.__dict__ = odeDict

        self.__neq_config = NEQConfig()
        neqDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/NEQConfig.json'), 'r'))
        self.__neq_config.__dict__ = neqDict

        self.__adjust_config = AdjustOrbitConfig()
        adjustOrbitDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/AdjustOrbitConfig.json'), 'r'))
        self.__adjust_config.__dict__ = adjustOrbitDict

        self.__interface_config = InterfaceConfig()
        interfaceDict = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/InterfaceConfig.json'), 'r'))
        self.__interface_config.__dict__ = interfaceDict

        self.__design_parameter_config = ParameterConfig()
        designParameter = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/DesignParameterConfig.json'), 'r'))
        self.__design_parameter_config.__dict__ = designParameter

        self.__frame_config = FrameConfig()
        frame = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/FrameConfig.json'), 'r'))
        self.__frame_config.__dict__ = frame

        self.__calibrateOrbit_config = CalibrateOrbitConfig()
        calibrateOrbit = json.load(open(os.path.abspath(self.__parent_path + '/setting/demo_3/CalibrateOrbitConfig.json'), 'r'))
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

        isPreprocess = StepControl.isPreprocess
        isAdjustOrbit = StepControl.isAdjustOrbit
        isAdjustKBRR = StepControl.isAdjustKBRR
        isGravityDesignMat = StepControl.isGetGravityDesignMat
        isNEQ = StepControl.isGetNEQ
        isSH = StepControl.isGetSH

        AdjustOrbitProcess = ParallelControl.AdjustOrbitProcess
        AdjustKBRRProcess = ParallelControl.AdjustKBRRProcess
        GetGravityDesignMatProcess = ParallelControl.GetGravityDesignMatProcess
        GetNEQProcess = ParallelControl.GetNEQProcess

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
        if isAdjustKBRR:
            self.makeLog(stepName='adjustKBRR', process=AdjustKBRRProcess)
            pool = multiprocessing.Pool(processes=AdjustKBRRProcess)
            pool.map(self._adjustKBRR, self.__arclist)
            pool.close()
            pool.join()
        if isGravityDesignMat:
            self.makeLog(stepName='gravityDesignMat', process=GetGravityDesignMatProcess)
            pool = multiprocessing.Pool(processes=GetGravityDesignMatProcess)
            pool.map(self._gravityDesignMat, self.__arclist)
            pool.close()
            pool.join()
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

    def _adjustOrbit(self, i):
        """step 5: init AdjustOrbit"""
        AdjustOrbit(accConfig=[self.__acc_config_A, self.__acc_config_B], date_span=self.__date_span, sat=self.__sat).\
            configure(SerDer=Orbit_var_2nd_diff, arcNo=i, ODEConfig=self.__ode_config,
                      FMConfig=self.__force_model_config, AdjustOrbitConfig=self.__adjust_config,
                      ParameterConfig=self.__parameter_config, InterfaceConfig=self.__interface_config,
                      FrameConfig=self.__frame_config).calibrate(iteration=2)
        pass

    def _adjustKBRR(self, j):
        RR = RangeRate(arcNo=j, date_span=self.__date_span)
        RR.configure(SerDer=Orbit_var_2nd_diff, AdjustOrbitConfig=self.__adjust_config,
                     FMConfig=self.__force_model_config, InterfaceConfig=self.__interface_config,
                     ParameterConfig=self.__parameter_config, FrameConfig=self.__frame_config,
                     ODEConfig=self.__ode_config)
        RR.calibrate()
        pass

    def _gravityDesignMat(self, k):
        dm = GravityDesignMat(arcNo=k, date_span=self.__date_span, sat=self.__sat).\
            configure(accConfig=[self.__acc_config_A, self.__acc_config_B],
                      ODEConfig=self.__ode_config, AdjustOrbitConfig=self.__adjust_config,
                      solverConfig=self.__solver_config, FMConfig=self.__force_model_config,
                      parameterConfig=self.__design_parameter_config, InterfaceConfig=self.__interface_config,
                      FrameConfig=self.__frame_config)
        dm.orbit_integration()
        dm.get_orbit_design_matrix()
        dm.get_sst_design_matrix()
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

    def __saveOrbitRMS(self):
        self.AdjustPath = AdjustOrbitConfig.PathOfFiles()
        self.AdjustPath.__dict__.update(self.__adjust_config.PathOfFilesConfig.copy())
        res_dir = pathlib.Path().joinpath(self.AdjustPath.StateVectorDataTemp, self.__date_span[0] + '_' + self.__date_span[1])
        interface_config = self.__interface_config
        rd = ArcData(interfaceConfig=interface_config)
        RMSAX = []
        RMSAY = []
        RMSAZ = []
        RMSBX = []
        RMSBY = []
        RMSBZ = []
        for i in self.__arclist:
            res_filename = res_dir.joinpath(str(i) + '_AB.hdf5')
            h5 = h5py.File(res_filename, 'r')
            rAB = h5['r'][:]
            rA = rAB[:, 0:3, 0]
            rB = rAB[:, 3:6, 0]
            gnva = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat='A')[:, 1:4]
            gnvb = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat='B')[:, 1:4]

            if np.shape(gnva) == np.shape(rA):
                diff = gnva - rA
                RMSAX.append(self.__rms(diff[:, 0]))
                RMSAY.append(self.__rms(diff[:, 1]))
                RMSAZ.append(self.__rms(diff[:, 2]))

            if np.shape(gnvb) == np.shape(rB):
                diff = gnvb - rB
                RMSBX.append(self.__rms(diff[:, 0]))
                RMSBY.append(self.__rms(diff[:, 1]))
                RMSBZ.append(self.__rms(diff[:, 2]))

        # np.save('OrbitRMSY.npy', RMSX)
        # np.save('OrbitRMSY.npy', RMSY)
        # np.save('OrbitRMSZ.npy', RMSZ)
        # X = np.load('OrbitRMSX.npy')
        # Y = np.load('OrbitRMSY.npy')
        # Z = np.load('OrbitRMSZ.npy')
        self.__plotOrbitRMS(RMSAX, RMSAY, RMSAZ, 'A')
        self.__plotOrbitRMS(RMSBX, RMSBY, RMSBZ, 'B')

        pass

    def __plotOrbitRMS(self,RMSX, RMSY, RMSZ, sat):
        plt.figure(dpi=300, figsize=(16, 8))
        plt.subplot(311)
        plt.title('Orbit-RMS-{}-{}-{}'.format(sat, self.__date_span[0], self.__date_span[-1]), fontsize=24)
        plt.plot(np.arange(len(RMSX)), [i * 1000 for i in RMSX], marker='o')
        plt.yticks(fontsize=18)
        plt.ylabel(r'$X/mm$', fontsize=20)

        plt.subplot(312)
        plt.plot(np.arange(len(RMSY)), [i * 1000 for i in RMSY], marker='o')
        plt.yticks(fontsize=18)
        plt.ylabel(r'$Y/mm$', fontsize=20)

        plt.subplot(313)
        plt.plot(np.arange(len(RMSZ)), [i * 1000 for i in RMSZ], marker='o')
        plt.xlabel('arcNo', fontsize=20)
        plt.ylabel(r'$Z/mm$', fontsize=20)
        plt.yticks(fontsize=18)

        plt.savefig('../result/img/{}-{}-{}.png'.format(sat, self.__date_span[0], self.__date_span[-1]))
        plt.grid(ls='--')
        plt.clf()
        pass

    def __saveKBRRRMS(self):
        self.AdjustPath = AdjustOrbitConfig.PathOfFiles()
        self.AdjustPath.__dict__.update(self.__adjust_config.PathOfFilesConfig.copy())

        res_dir = pathlib.Path().joinpath(self.AdjustPath.RangeRateTemp, self.__date_span[0] + '_' + self.__date_span[1])
        post_RMS = []
        for i in self.__arclist:
            res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
            if os.path.exists(res_filename):
                h5 = h5py.File(res_filename, 'r')
                if len(h5.keys()) != 0:
                    postfit = h5['post_residual'][()]
                    post_RMS.append(self.__rms(postfit))
        plt.figure(dpi=300, figsize=(16, 8))
        plt.title('KBRR-RMS-{}-{}'.format(self.__date_span[0], self.__date_span[-1]), fontsize=24)
        plt.plot(np.arange(len(post_RMS)), [i * 1000000 for i in post_RMS], marker='o', label="arc RMS")
        plt.yticks(fontsize=18)
        plt.ylabel(r'$X/um$', fontsize=20)
        plt.legend(loc='upper right', fontsize=20)
        plt.savefig('../result/img/KBRR-{}-{}.png'.format(self.__date_span[0], self.__date_span[-1]))
        pass

    def __rms(self, x: np.ndarray):
        """
        :param x: 1-dim
        :return:
        """
        # 确认公式
        return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])

    def getSpaceKBRR(self):

        frame_config = FrameConfig()
        frame = json.load(open(('../setting/Calibrate/FrameConfig.json'), 'r'))
        frame_config.__dict__ = frame
        eop = EOP().configure(frameConfig=frame_config).load()
        fr = Frame(eop).configure(frameConfig=frame_config)
        interface_config = InterfaceConfig()
        interfaceDict = json.load(open('../setting/Calibrate/InterfaceConfig.json', 'r'))
        interface_config.__dict__ = interfaceDict
        rd = ArcData(interfaceConfig=interface_config)
        Lon = []
        Lat = []
        Postfit = []
        RMS = []

        date_span = interface_config.date_span
        sat = interface_config.sat
        date_list = GeoMathKit.getEveryDay(*date_span)
        days = datetime.strptime(date_list[-1], '%Y-%m-%d') - datetime.strptime(date_list[0], '%Y-%m-%d')
        arcLen = days.days * (24 / interface_config.arc_length)

        for i in range(0, int(arcLen)):
            if i != 110:
                GNV = rd.getData(arc=i, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
                GNVTime = GNV[:, 0]
                GNVR = GNV[:, 1:4]
                res_dir = pathlib.Path('../temp/RangeRate').joinpath(date_span[0] + '_' + date_span[1])
                res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
                h5 = h5py.File(res_filename, 'r')
                t2 = h5['post_t'][()]
                Postfit.extend(h5['post_residual'][()])
                postfit = h5['post_residual'][()]

                RMS.append(self.__rms(postfit))

                new_index = [list(GNVTime).index(x) for x in t2]
                new_Time = GNVTime[new_index]
                new_GNVR = GNVR[new_index]
                for i in range(len(new_Time)):
                    time = new_Time[i]
                    pos = new_GNVR[i]
                    fr = fr.setTime(float(time))
                    itrs = fr.PosGCRS2ITRS(np.array(pos).astype(float), fr.getRotationMatrix)
                    lon, lat, r = GeoMathKit.CalcPolarAngles(itrs)
                    Lon.append(lon)
                    Lat.append(lat)

        np.savez('ResKBRR.npz', data=RMS)
        np.savez('SpaceKBRR.npz', x=Lon, y=Lat, z=Postfit)

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
    demo3 = demo_3()
    demo3.loadJson()
    demo3.run()
    pass