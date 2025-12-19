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
from src.Auxilary.Outlier import Outlier
from src.Preference.EnumType import Payload, SatID, SSTObserve
from scipy.signal import butter, filtfilt
from scipy import signal
from scipy.linalg import toeplitz, cholesky, eigh, solve_triangular
from scipy.sparse.linalg import eigsh
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d
from scipy.fft import dct, idct
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import block_diag


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
        self.lagNum = None
        pass

    def loadJson(self):
        self.__acc_config_A = AccelerometerConfig()
        acc_A = json.load(open(os.path.join(self.__parent_path, 'setting/Calibrate/AccelerometerConfig_A.json'), 'r'))
        self.__acc_config_A.__dict__ = acc_A

        self.__acc_config_B = AccelerometerConfig()
        acc_B = json.load(open(os.path.abspath(self.__parent_path + '/setting/Calibrate/AccelerometerConfig_B.json'), 'r'))
        self.__acc_config_B.__dict__ = acc_B

        self.__design_acc_config_A = AccelerometerConfig()
        designacc_A = json.load(open(os.path.join(self.__parent_path, 'setting/Calibrate/DesignAccelerometerConfig_A.json'), 'r'))
        self.__design_acc_config_A.__dict__ = designacc_A

        self.__design_acc_config_B = AccelerometerConfig()
        designacc_B = json.load(
            open(os.path.abspath(self.__parent_path + '/setting/Calibrate/DesignAccelerometerConfig_B.json'), 'r'))
        self.__design_acc_config_B.__dict__ = designacc_B

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
            self.__saveOrbitRMS()
        if isAdjustKBRR:
            self.makeLog(stepName='adjustKBRR', process=AdjustKBRRProcess)
            pool = multiprocessing.Pool(processes=AdjustKBRRProcess)
            pool.map(self._adjustKBRR, self.__arclist)
            pool.close()
            pool.join()
            self.__saveKBRRRMS()
            # self.butter_low_filter()
        if isGravityDesignMat:
            self.makeLog(stepName='gravityDesignMat', process=GetGravityDesignMatProcess)
            # for i in np.arange(0, 10, 1):
            #     self._gravityDesignMat(i)
            pool = multiprocessing.Pool(processes=GetGravityDesignMatProcess)
            pool.map(self._gravityDesignMat, self.__arclist)
            pool.close()
            pool.join()
        if isNEQ:
            self.makeLog(stepName='NEQ', process=GetNEQProcess)
            pool = multiprocessing.Pool(processes=GetNEQProcess)
            pool.map(self._NEQ, self.__arclist)
            pool.close()
            pool.join()
        if isSH:
            # self.makeLog(stepName='SH', process=1)
            # self.__calculate_cs(int(self.__solver_config.OrbitKinFactor))
            # self.plotPostKbrr()
            # self.xyz2lonlat()

            # self.GLS()
            #
            self.makeLog(stepName='NEQ', process=GetNEQProcess)
            pool = multiprocessing.Pool(processes=GetNEQProcess)
            pool.map(self._NEQ, self.__arclist)
            pool.close()
            pool.join()
            # #
            self.makeLog(stepName='SH', process=1)
            self.__calculate_cs(int(1))
            # self.plotPostKbrr()
            # self.xyz2lonlat()
            # self.re_calculate_cs()
        # self.getSpaceKBRR()
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
            configure(accConfig=[self.__design_acc_config_A, self.__design_acc_config_B],
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

    def __calculate_cs(self, OrbitKinFactor):
        '''orbit and kbrr factor'''
        neq_path_config = self.__neq_config.PathOfFiles()
        neq_path_config.__dict__.update(self.__neq_config.PathOfFilesConfig.copy())
        interface_config = self.__interface_config
        stokes_coefficients_config = self.__design_parameter_config.StokesCoefficients()
        stokes_coefficients_config.__dict__.update(self.__design_parameter_config.StokesCoefficientsConfig.copy())

        self.AccelerometerConfig = self.__design_parameter_config.Accelerometer()
        self.AccelerometerConfig.__dict__.update(self.__design_parameter_config.AccelerometerConfig.copy())

        '''config force model path'''
        force_model_path_config = self.__force_model_config.PathOfFiles()
        force_model_path_config.__dict__.update(self.__force_model_config.PathOfFilesConfig.copy())

        res_dir1 = neq_path_config.NormalEqTemp + '/' +\
                  interface_config.date_span[0] + '_' + interface_config.date_span[1]

        # res_dir2 = neq_path_config.NormalEqTemp + '/' + \
        #            interface_config.date_span[0] + '_' + interface_config.date_span[1] + '-onebyone'

        result_name = neq_path_config.ResultCS + '/' + \
                      interface_config.date_span[0] + '_' + interface_config.date_span[1] + '.hdf5'

        parameter_number = stokes_coefficients_config.Parameter_Number

        if self.AccelerometerConfig.isScale:
            parameter_number += 18

        NMatrix1 = np.zeros((parameter_number, parameter_number))
        LMatrix1 = np.zeros((parameter_number, 1))

        date_span = interface_config.date_span

        self.n = interface_config.arc_length * 3600 / interface_config.sr_target * 3
        self.p = parameter_number

        for i in tqdm(self.__arclist, desc='Solve normal equations: '):
        # for i in tqdm((0, 1), desc='Solve normal equations: '):
            res_filename1 = res_dir1 + '/' + str(i) + '.hdf5'
            h51 = h5py.File(res_filename1, 'r')
            Orbit_N_A = h51['Orbit_N_A'][:]
            Orbit_N_B = h51['Orbit_N_B'][:]
            RangeRate_N = h51['RangeRate_N'][:]

            Orbit_l_A = h51['Orbit_l_A'][:]
            Orbit_l_B = h51['Orbit_l_B'][:]
            RangeRate_l = h51['RangeRate_l'][:]
            # if self.AccelerometerConfig.isScale:
                # Orbit_N_A, Orbit_l_A = self.keepGlobal(Orbit_N_A, Orbit_l_A)
                # Orbit_N_B, Orbit_l_B = self.keepGlobal(Orbit_N_B, Orbit_l_B)
                # RangeRate_N, RangeRate_l = self.keepGlobal(RangeRate_N, RangeRate_l)

            if self.AccelerometerConfig.isScale:
                NMatrix1[0:9, 0:9] = NMatrix1[0:9, 0:9] + Orbit_N_A[0:9, 0:9] + RangeRate_N[0:9, 0:9] * OrbitKinFactor
                NMatrix1[0:9, 18:] = NMatrix1[0:9, 18:] + Orbit_N_A[0:9, 9:] + RangeRate_N[0:9, 18:] * OrbitKinFactor

                NMatrix1[9:18, 9:18] = NMatrix1[9:18, 9:18] + Orbit_N_B[0:9, 0:9] + RangeRate_N[9:18, 9:18] * OrbitKinFactor
                NMatrix1[9:18, 18:] = NMatrix1[9:18, 18:] + Orbit_N_B[0:9, 9:] + RangeRate_N[9:18, 18:] * OrbitKinFactor

                NMatrix1[18:, 0:9] = NMatrix1[18:, 0:9] + Orbit_N_A[9:, 0:9] + RangeRate_N[18:, 0:9] * OrbitKinFactor
                NMatrix1[18:, 9:18] = NMatrix1[18:, 9:18] + Orbit_N_B[9:, 0:9] + RangeRate_N[18:, 9:18] * OrbitKinFactor

                NMatrix1[18:, 18:] = NMatrix1[18:, 18:] + Orbit_N_A[9:, 9:] + Orbit_N_B[9:, 9:] + RangeRate_N[18:, 18:] * OrbitKinFactor

                LMatrix1[0:9] = LMatrix1[0:9] + Orbit_l_A[0:9] + RangeRate_l[0:9] * OrbitKinFactor
                LMatrix1[9:18] = LMatrix1[9:18] + Orbit_l_B[0:9] + RangeRate_l[9:18] * OrbitKinFactor
                LMatrix1[18:] = LMatrix1[18:] + Orbit_l_A[9:] + Orbit_l_B[9:] + RangeRate_l[18:] * OrbitKinFactor
            else:
                NMatrix1 = NMatrix1 + (Orbit_N_A + Orbit_N_B) + RangeRate_N * OrbitKinFactor
                LMatrix1 = LMatrix1 + (Orbit_l_A + Orbit_l_B) + RangeRate_l * OrbitKinFactor
            h51.close()

            # res_filename2 = res_dir2 + '/' + str(i) + '.hdf5'
            # h52 = h5py.File(res_filename2, 'r')
            # NMatrix2 = NMatrix2 + (h52['Orbit_N_A'][:] + h52['Orbit_N_B'][:]) + h52['RangeRate_N'][:] * OrbitKinFactor
            # LMatrix2 = LMatrix2 + (h52['Orbit_l_A'][:] + h52['Orbit_l_B'][:]) + h52['RangeRate_l'][:].reshape(
            #     (parameter_number, 1)) * OrbitKinFactor
            # h52.close()

        cs = np.linalg.lstsq(NMatrix1, LMatrix1[:, 0], rcond=None)[0]

        # cs2 = np.linalg.lstsq(NMatrix2, LMatrix2[:, 0], rcond=None)[0]
        if self.AccelerometerConfig.isScale:
            scale = cs[0:18]
            print(scale)
            cs = cs[18:]

        sigma = self.getLocalPar(solverConfig=self.__solver_config, cs=cs, N=NMatrix1)
        # sigma = cs

        sort = SortCS(method=stokes_coefficients_config.SortMethod,
                      degree_max=stokes_coefficients_config.MaxDegree,
                      degree_min=stokes_coefficients_config.MinDegree)
        c, s = sort.invert(cs)
        sigmaC , sigmaS = sort.invert(sigma)

        static_c, static_s = LoadGif48().load(force_model_path_config.Gif48).getCS(stokes_coefficients_config.MaxDegree)

        C, S = c + static_c, s + static_s

        format = FormatWrite().configure(filedir=neq_path_config.ResultCS,
                                         data=date_span, degree=stokes_coefficients_config.MaxDegree,
                                         c=C, s=S, sigmaC=sigmaC, sigmaS=sigmaS)
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

        if not os.path.exists('../result/img'):
            os.mkdir('../result/img')
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
        plt.ylim(0.2, 0.5)
        plt.ylabel(r'$X/um$', fontsize=20)
        plt.legend(loc='upper right', fontsize=20)
        plt.savefig('../result/img/KBRR-{}-{}.png'.format(self.__date_span[0], self.__date_span[-1]))
        plt.clf()
        pass

    # def butter_low_filter(self):
    #     self.AdjustPath = AdjustOrbitConfig.PathOfFiles()
    #     self.AdjustPath.__dict__.update(self.__adjust_config.PathOfFilesConfig.copy())
    #
    #     res_dir = pathlib.Path().joinpath(self.AdjustPath.RangeRateTemp,
    #                                       self.__date_span[0] + '_' + self.__date_span[1])
    #
    #     timespan_arc = {}
    #     post_time = {}
    #     post_arc = []
    #     post_design_matrix = {}
    #     for i in self.__arclist:
    #         res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
    #         if os.path.exists(res_filename):
    #             h5 = h5py.File(res_filename, 'r')
    #             if len(h5.keys()) != 0:
    #                 time = h5['post_t'][()]
    #                 post_time[i] = time
    #                 timespan_arc[i] = len(time)
    #                 post_design_matrix[i] = h5['design_matrix'][()]
    #                 postfit = h5['post_residual'][()]
    #                 post_arc.extend(list(postfit))
    #             h5.close()
    #
    #     fs = 0.2
    #     cutoff = 0.005
    #     nyq = 0.5 * fs
    #     normal_cutoff = cutoff / nyq
    #     b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    #     filtered_data = filtfilt(b, a, post_arc)
    #
    #     index = 0
    #     filtered_dir = pathlib.Path().joinpath(self.AdjustPath.FilteredRangeRateTemp,
    #                                       self.__date_span[0] + '_' + self.__date_span[1])
    #     for i in self.__arclist:
    #         if not os.path.exists(filtered_dir):
    #             os.makedirs(filtered_dir)
    #         res_filename = filtered_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
    #         h5 = h5py.File(res_filename, 'w')
    #
    #         h5.create_dataset('post_t', data=post_time[i])
    #         h5.create_dataset('post_residual', data=filtered_data[index: index + timespan_arc[i]])
    #         h5.create_dataset('design_matrix', data=post_design_matrix[i])
    #         h5.close()
    #
    #         index += timespan_arc[i]
    #
    #     npe = int(fs * len(post_arc))
    #     frequencies_welch_post_arc, psd_welch_post_arc = signal.welch(post_arc, fs, nperseg=npe)
    #     frequencies_filtered, psd_filtered = signal.welch(filtered_data, fs, nperseg=npe)
    #     plt.plot(frequencies_welch_post_arc, psd_welch_post_arc, lw=1.5, alpha=0.8, color='black', label='Non')
    #     plt.plot(frequencies_filtered, psd_filtered, lw=1.5, alpha=0.8, color='red', label=cutoff)
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     plt.grid(True, linestyle='dashed')
    #     font = {'size': 20}
    #     plt.xlabel('Frequency [Hz]', fontdict=font)
    #     plt.ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
    #     plt.legend(loc=2, fontsize='20')
    #     plt.savefig('../result/img/psd.png'.format(self.__date_span[0], self.__date_span[-1]))
    #
    #     return self

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

    def getLocalPar(self, solverConfig, cs, N):
        self._AdjustPathConfig = self.__adjust_config.PathOfFiles()
        self._AdjustPathConfig.__dict__.update(self.__adjust_config.PathOfFilesConfig.copy())
        self.cs=cs
        '''get orbit par'''
        res_dir = solverConfig.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]

        self.b_rss = 0

        for i in tqdm(self.__arclist, desc='Solve local parameters: '):
            res_filename = res_dir + '/Orbit_' + str(i) + '_' + 'AB.hdf5'
            h5 = h5py.File(res_filename, 'r')
            self.__orbit_nominal = h5['r'][()]
            self.__t_nominal = h5['t'][()]
            self.__local_r = h5['local_r'][()]
            self.__global_r = h5['global_r'][()]
            h5.close()

            LocalA = self.getOrbitPar(sat=SatID.A, arcNo=i)
            LocalB = self.getOrbitPar(sat=SatID.B, arcNo=i)

            self.saveAccPar(LocalA, LocalB, arcNo=i)
            self.savePostKbrr(arcNo=i)

        sigma2_hat = self.b_rss / (self.n - self.p)
        cov = sigma2_hat * np.linalg.inv(N)
        sigma = np.sqrt(np.diag(cov))

        CovName = solverConfig.CovarianceMatrix + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '.hdf5'
        if not os.path.exists(solverConfig.CovarianceMatrix):
            os.mkdir(solverConfig.CovarianceMatrix)
        h5 = h5py.File(CovName, 'w')
        h5.create_dataset('cov', data=cov)
        h5.close()

        return sigma

    def getOrbitPar(self, sat: SatID, arcNo: int):
        if sat == SatID.A:
            orbit_nominal = self.__orbit_nominal[:, 0:3]
            local_r = self.__local_r[:, 0:3, :]
            global_r = self.__global_r[:, 0:3, :]
        else:
            orbit_nominal = self.__orbit_nominal[:, 3:6]
            local_r = self.__local_r[:, 3:6, :]
            global_r = self.__global_r[:, 3:6, :]

        t_nominal = self.__t_nominal
        self._rd = ArcData(interfaceConfig=self.__interface_config)
        orbit_obs = self._rd.getData(arc=arcNo, kind=Payload.GNV, sat=sat)[:, 0:4]
        #
        # orbit_obs = self._rd.getData(arc=self.__arclist[arcNo], kind=Payload.KinematicOrbit, sat=sat)[:, 0:4]
        # orbit_obs = self.kin_obs_outlier(sat=sat, kin_orbit=orbit_obs, arc=self.__arclist[arcNo])

        t_obs = orbit_obs[:, 0]
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
        '''Expanding the observations'''
        obs = orbit_obs[:, 1:] - orbit_nominal

        obs = obs.reshape((-1, 1))

        # TODO: Fix the kinematic orbit

        '''Expanding the design matrix'''
        n_local = np.shape(local_r)[-1]
        n_global = np.shape(global_r)[-1]
        local_r = local_r.reshape((-1, n_local))
        global_r = global_r.reshape((-1, n_global))

        Local = np.linalg.lstsq(local_r, (obs[:, 0] - global_r @ self.cs), rcond=None)[0]

        b_res = obs[:, 0] - local_r @ Local - global_r @ self.cs

        self.b_rss += np.sum(b_res ** 2)

        res_path = self.__solver_config.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'
        if not os.path.exists(res_path):
            os.makedirs(res_path, exist_ok=True)
        res_name = res_path + 'Orbit_' + str(arcNo) + str(sat.name) + '.hdf5'
        h5 = h5py.File(res_name, 'w')
        h5.create_dataset('post_Orbit', data=b_res)
        h5.close()
        return Local

    def saveAccPar(self, par_a, par_b, arcNo: int):
        res_path = self.__solver_config.PostParameter + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'
        if not os.path.exists(res_path):
            os.makedirs(res_path, exist_ok=True)

        res_name = res_path + str(arcNo) + '_AB.hdf5'
        h5 = h5py.File(res_name, 'w')
        h5.create_dataset('par_a', data=par_a)
        h5.create_dataset('par_b', data=par_b)
        h5.close()
        pass

    def savePostKbrr(self, arcNo: int):
        '''get kbrr par'''
        dm_dir = self.__solver_config.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = dm_dir + '/' + SSTObserve.RangeRate.name + '_' + str(arcNo) + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        t_dm = h5['t'][()]
        global_dm = h5['global'][()]
        local_dm = h5['local'][()]
        h5.close()

        dm_dir = self._AdjustPathConfig.RangeRateTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = dm_dir + '/' + SSTObserve.RangeRate.name + '_' + str(arcNo) + '.hdf5'
        h5 = h5py.File(res_filename, 'r')
        t_obs = h5['post_t'][()]
        obs = h5['post_residual'][()]
        local_dm_empirical = h5['design_matrix'][()]
        h5.close()

        int_t_obs = t_obs.astype(np.int64)
        int_t_dm = t_dm.astype(np.int64)
        new_index = [list(int_t_dm).index(x) for x in int_t_obs]

        '''Reassignment'''
        t_dm = t_dm[new_index]
        global_dm = global_dm[new_index, :]
        local_dm = local_dm[new_index, :]

        Local_par = np.linalg.lstsq(local_dm, (obs - global_dm @ self.cs))[0]
        kbrr_res = obs - local_dm @ Local_par - global_dm @ self.cs

        res_path = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'
        if not os.path.exists(res_path):
            os.makedirs(res_path, exist_ok=True)

        res_name = res_path + 'RangeRate_' + str(arcNo) + '.hdf5'
        h5 = h5py.File(res_name, 'w')
        h5.create_dataset('kbrr_par', data=Local_par)
        h5.create_dataset('post_kbrr', data=kbrr_res)
        h5.close()
        pass

    def plotPostKbrr(self):
        self.AdjustPath = AdjustOrbitConfig.PathOfFiles()
        self.AdjustPath.__dict__.update(self.__adjust_config.PathOfFilesConfig.copy())

        res_dir = pathlib.Path().joinpath(self.AdjustPath.RangeRateTemp,
                                          self.__date_span[0] + '_' + self.__date_span[1])

        post_dir = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1]
        pre_RMS = []
        post_RMS = []
        for i in self.__arclist:
            res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
            post_filename = post_dir + '/RangeRate_' + str(i) + '.hdf5'
            if os.path.exists(res_filename):
                h5 = h5py.File(res_filename, 'r')
                post_h5 = h5py.File(post_filename, 'r')
                if len(h5.keys()) != 0:
                    postfit = h5['post_residual'][()]
                    pre_RMS.append(self.__rms(postfit))

                    post_kbrr = post_h5['post_kbrr'][()]
                    post_RMS.append(self.__rms(post_kbrr))
                h5.close()
                post_h5.close()

        plt.figure(dpi=300, figsize=(16, 8))
        plt.title('KBRR-RMS-{}-{}'.format(self.__date_span[0], self.__date_span[-1]), fontsize=24)
        plt.plot(np.arange(len(pre_RMS)), [i * 1000000 for i in pre_RMS], marker='o', color='green', label="pre kbrr RMS")
        plt.plot(np.arange(len(post_RMS)), [i * 1000000 for i in post_RMS], marker='*', color='red', label="post kbrr RMS")
        plt.yticks(fontsize=18)
        plt.ylim(0, 0.7)
        plt.ylabel(r'$X/um$', fontsize=20)
        plt.legend(loc='upper right', fontsize=20)

        if not os.path.exists('../result/img'):
            os.mkdir('../result/img')

        plt.savefig('../result/img/post-KBRR-{}-{}.png'.format(self.__date_span[0], self.__date_span[-1]))
        pass

    def xyz2lonlat(self):
        datespan = self.__date_span

        eop = EOP().configure(frameConfig=self.__frame_config).load()
        fr = Frame(eop).configure(frameConfig=self.__frame_config)
        self.AdjustPath = AdjustOrbitConfig.PathOfFiles()
        self.AdjustPath.__dict__.update(self.__adjust_config.PathOfFilesConfig.copy())
        raw_data_path = self.InterfacePathConfig.temp_raw_data
        filename = pathlib.Path().joinpath(raw_data_path, datespan[0] + '_' + datespan[1] + ".hdf5")
        post_dir = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1]

        res_dir = pathlib.Path().joinpath(self.AdjustPath.RangeRateTemp,
                                          self.__date_span[0] + '_' + self.__date_span[1])

        h5data = h5py.File(filename, "r")

        Lon = []
        Lat = []
        Postfit = []
        Prefit = []
        for j in self.__arclist:
            GNV = h5data['GNV_A_arc' + str(j)][()]
            GNVTime = GNV[:, 0]
            GNVR = GNV[:, 1:4]

            post_filename = post_dir + '/RangeRate_' + str(j) + '.hdf5'
            res_filename = res_dir.joinpath('RangeRate_' + str(j) + '.hdf5')
            if not os.path.exists(res_filename):
                continue
            h5 = h5py.File(res_filename, 'r')
            post_h5 = h5py.File(post_filename, 'r')

            t2 = h5['post_t'][()]

            commonTime = set(GNVTime.astype(np.int64))
            commonTime = commonTime & set(t2.astype(np.int64))
            commonTime = np.array(list(commonTime))
            commonTime.sort()

            new_index1 = [list(GNVTime).index(x) for x in commonTime]
            new_index2 = [list(t2).index(x) for x in commonTime]
            new_Time = GNVTime[new_index1]
            new_GNVR = GNVR[new_index1]
            Prefit.extend(h5['post_residual'][()][new_index2])
            Postfit.extend(post_h5['post_kbrr'][()][new_index2])

            for k in range(len(new_Time)):
                time = new_Time[k]
                pos = new_GNVR[k]
                fr = fr.setTime(float(time))
                itrs = fr.PosGCRS2ITRS(np.array(pos).astype(float), fr.getRotationMatrix)
                lon, lat, r = GeoMathKit.CalcPolarAngles(itrs)
                Lon.append(lon)
                Lat.append(lat)
            h5.close()

        Lon = np.degrees(np.array(Lon))
        Lat = np.degrees(np.array(Lat))
        Prefit = np.array(Prefit)
        Postfit = np.array(Postfit)

        pre_filename = '../result/pre-' + datespan[0] + '_' + datespan[1] + '.hdf5'
        pre_h5 = h5py.File(pre_filename, "w")
        pre_h5.create_dataset('lon', data=Lon)
        pre_h5.create_dataset('lat', data=Lat)
        pre_h5.create_dataset('prefit', data=Prefit)
        pre_h5.close()

        post_filename = '../result/post-' + datespan[0] + '_' + datespan[1] + '.hdf5'
        post_h5 = h5py.File(post_filename, "w")
        post_h5.create_dataset('lon', data=Lon)
        post_h5.create_dataset('lat', data=Lat)
        post_h5.create_dataset('postfit', data=Postfit)
        post_h5.close()

        pass

    def re_calculate_cs(self):
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

        self.n = interface_config.arc_length * 3600 / interface_config.sr_target * 3
        self.p = parameter_number

        PostKBRR_path = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'
        PostOrbit_path = self.__solver_config.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'

        for i in tqdm(self.__arclist, desc='Re Solve normal equations: '):
            PostKBRR_name = PostKBRR_path + 'RangeRate_' + str(i) + '.hdf5'
            resA_name = PostOrbit_path + 'Orbit_' + str(i) + 'A.hdf5'
            resB_name = PostOrbit_path + 'Orbit_' + str(i) + 'B.hdf5'

            PostKBRR_h5 = h5py.File(PostKBRR_name, 'r')
            post_kbrr = PostKBRR_h5['post_kbrr'][:]
            kbrr = self.__rms(post_kbrr)

            resA_name_h5 = h5py.File(resA_name, 'r')
            post_Orbit_A = resA_name_h5['post_Orbit'][:]
            leo_A = self.__rms(post_Orbit_A) * 10
            # leo_A = 0.02

            resB_name_h5 = h5py.File(resB_name, 'r')
            post_Orbit_B = resB_name_h5['post_Orbit'][:]
            leo_B = self.__rms(post_Orbit_B) * 10
            # leo_B = 0.02

            res_filename = res_dir + '/' + str(i) + '.hdf5'
            h5 = h5py.File(res_filename, 'r')
            NMatrix = NMatrix + (1 / (leo_A ** 2) * h5['Orbit_N_A'][:] + 1 / (kbrr ** 2) * h5['RangeRate_N'][:]
                                 + 1 / (leo_B ** 2) * h5['Orbit_N_B'][:])

            LMatrix = LMatrix + (1 / (leo_A ** 2) * h5['Orbit_l_A'][:] + 1 / (leo_B ** 2) * h5['Orbit_l_B'][:]
                                 + 1 / (kbrr ** 2) * h5['RangeRate_l'][:].reshape((parameter_number, 1)))

            h5.close()
            PostKBRR_h5.close()
            resA_name_h5.close()
            resB_name_h5.close()

        cs = np.linalg.lstsq(NMatrix, LMatrix[:, 0], rcond=None)[0]

        sigma = self.getLocalPar(solverConfig=self.__solver_config, cs=cs, N=NMatrix)
        # QA, RA = np.linalg.qr(NMatrix)
        # cs = np.linalg.inv(RA).dot(QA.T).dot(LMatrix[:, 0])

        # cs = []
        # with open('../result/dcs.txt', 'r') as file:
        #     lines = file.readlines()
        #     # 创建一个列表来存储转换后的浮点数
        #     for line in lines:
        #         try:
        #             number = float(line.strip())
        #             cs.append(number)
        #         except ValueError:
        #             print(f"无法将'{line.strip()}'转换为浮点数")

        sort = SortCS(method=stokes_coefficients_config.SortMethod,
                      degree_max=stokes_coefficients_config.MaxDegree,
                      degree_min=stokes_coefficients_config.MinDegree)
        c, s = sort.invert(cs)
        sigmaC , sigmaS = sort.invert(sigma)

        static_c, static_s = LoadGif48().load(force_model_path_config.Gif48).getCS(stokes_coefficients_config.MaxDegree)

        C, S = c + static_c, s + static_s

        format = FormatWrite().configure(filedir=neq_path_config.ResultCS,
                                         data=date_span, degree=stokes_coefficients_config.MaxDegree,
                                         c=C, s=S, sigmaC=sigmaC, sigmaS=sigmaS)
        format.unfiltered()

        pass

    def kin_obs_outlier(self, kin_orbit, sat, arc):
        self._AdjustConfig = self.__adjust_config.Orbit()
        self._AdjustConfig.__dict__.update(self.__adjust_config.OrbitConfig.copy())

        gnv_orbit = self._rd.getData(arc=arc, kind=Payload.GNV, sat=sat)[:, 0:4]
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

        gnv_orbit[index, 1:] = kin_orbit[index, 1:]

        return gnv_orbit

    def keepGlobal(self, N, L):
        N_ll = N[0:9, 0:9]
        N_lg = N[0:9, 9:]
        N_gl = N[9:, 0:9]
        N_gg = N[9:, 9:]

        l_l = L[0:9]
        l_g = L[9:]

        M_n = np.linalg.lstsq(N_ll, N_lg, rcond=None)[0]
        M_l = np.linalg.lstsq(N_ll, l_l, rcond=None)[0]

        N_G = N_gg - N_gl @ M_n
        N_l = l_g - N_gl @ M_l

        return N_G, N_l

    def GLS(self):
        '''get orbit obs'''
        orbit_path = self.__solver_config.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1]
        kbrr_path = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1]
        self.lagNum = int(self.__interface_config.arc_length * 3600 / self.__interface_config.sr_target)

        kbrr = []
        orbit_a = []
        orbit_b = []

        # acov_a_x = np.zeros((self.lagNum,))
        # acov_a_y = np.zeros((self.lagNum,))
        # acov_a_z = np.zeros((self.lagNum,))
        # acov_b_x = np.zeros((self.lagNum,))
        # acov_b_y = np.zeros((self.lagNum,))
        # acov_b_z = np.zeros((self.lagNum,))
        # acov_kbrr = np.zeros((self.lagNum,))
        for i in tqdm(self.__arclist, desc='Solve Orbit and KBRR Cov: '):
        # for i in tqdm((0, 1), desc='Solve Orbit and KBRR Cov: '):
            postOrbitname_a = orbit_path + '/Orbit_' + str(i) + 'A.hdf5'
            postOrbitname_b = orbit_path + '/Orbit_' + str(i) + 'B.hdf5'
            kbrrname = kbrr_path + '/' + SSTObserve.RangeRate.name + '_' + str(i) + '.hdf5'

            postOrbith5_a = h5py.File(postOrbitname_a, "r")
            postOrbith5_b = h5py.File(postOrbitname_b, "r")
            postKBRRh5 = h5py.File(kbrrname, "r")

            res_orbit_a = postOrbith5_a['post_Orbit'][:]
            res_orbit_b = postOrbith5_b['post_Orbit'][:]
            res_kbrr = postKBRRh5['post_kbrr'][:]

            # res_a = np.array(res_orbit_a).reshape((-1, 3))
            # res_a_x = res_a[:, 0]
            # res_a_y = res_a[:, 1]
            # res_a_z = res_a[:, 2]
            #
            # res_b = np.array(res_orbit_b).reshape((-1, 3))
            # res_b_x = res_b[:, 0]
            # res_b_y = res_b[:, 1]
            # res_b_z = res_b[:, 2]
            #
            # acov_a_x += self.estimate_acov(res_a_x, max_lag=len(res_a_x) - 1)
            # acov_a_y += self.estimate_acov(res_a_y, max_lag=len(res_a_x) - 1)
            # acov_a_z += self.estimate_acov(res_a_z, max_lag=len(res_a_x) - 1)
            #
            # acov_b_x += self.estimate_acov(res_b_x, max_lag=len(res_b_x) - 1)
            # acov_b_y += self.estimate_acov(res_b_y, max_lag=len(res_a_x) - 1)
            # acov_b_z += self.estimate_acov(res_b_z, max_lag=len(res_a_x) - 1)
            #
            # acov_kbrr += self.estimate_acov(res_kbrr, max_lag=len(res_kbrr) - 1)
            orbit_a.extend(res_orbit_a)
            orbit_b.extend(res_orbit_b)
            kbrr.extend(res_kbrr)

            postOrbith5_a.close()
            postOrbith5_b.close()
            postKBRRh5.close()

        # acov_a_x = acov_a_x / len(self.__arclist)
        # Sigma_a = self.getSigma6(acov_a_x, acov_a_y, acov_a_z)
        # Sigma_b = self.getSigma6(acov_b_x, acov_b_y, acov_b_z)
        # Sigma = self.getSigma5(acov_kbrr)

        Sigma_a = self.getSigma2(res=orbit_a)
        Sigma_b = self.getSigma2(res=orbit_b)
        Sigma = self.getSigma(res=kbrr)

        for i in tqdm(self.__arclist, desc='save Orbit and KBRR Cov: '):
        # for i in tqdm((0, 1), desc='save Orbit and KBRR Cov: '):
            kbrrname = kbrr_path + '/' + SSTObserve.RangeRate.name + '_' + str(i) + '.hdf5'
            postOrbitname_a = orbit_path + '/Orbit_' + str(i) + 'A.hdf5'
            postOrbitname_b = orbit_path + '/Orbit_' + str(i) + 'B.hdf5'

            postOrbith5_a = h5py.File(postOrbitname_a, "r+")
            postOrbith5_b = h5py.File(postOrbitname_b, "r+")
            postKBRRh5 = h5py.File(kbrrname, "r+")

            self.saveSigma1(Sigma_a, h5path=postOrbith5_a)
            self.saveSigma1(Sigma_b, h5path=postOrbith5_b)
            self.saveSigma1(Sigma, h5path=postKBRRh5)
        pass

    def getSigma(self, res):
        n = len(res)
        res = detrend(res, type="linear")
        res -= res.mean()
        # res = detrend(res)
        # res -= np.mean(res)

        # p_max = min(50, n // 10)
        # p = self.pick_AR_order(res, p_max=p_max, ic='bic')
        #
        # phi, sigma2 = self.estimate_ARp(res, p)
        # cov = self.AR_toeplitz_cov(phi, sigma2, n)

        # res = detrend(res, type='linear')
        acov = self.estimate_acov(res, max_lag=self.lagNum - 1)
        # FFT
        eps = max(acov[0] * 1e-3, 1e-16)
        # eps = max(acov[0] * 1e-8, 1e-20)
        # Step 1: DCT-I
        S = dct(acov, type=1, norm='ortho')
        # f = np.linspace(0, 1 / (2 * 5), len(S))
        # plt.figure(figsize=(7, 4))
        # plt.loglog(f[1:], S[1:], lw=1.5)
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("PSD (variance)")
        # plt.title("Noise PSD estimated via DCT-I")
        # plt.grid(True)
        # plt.axhline(eps, color='r', ls='--', label='eps')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        # Step 2: clipping
        S_clipped = np.maximum(S, eps)
        # Step 3: inverse DCT-I
        acov_corr = idct(S_clipped, type=1, norm='ortho')
        Sigma = toeplitz(acov_corr)
        return Sigma

    def getSigma2(self, res):
        res = np.array(res).reshape((-1, 3))
        res_x = res[:, 0]
        res_y = res[:, 1]
        res_z = res[:, 2]

        res_x = detrend(res_x, type="linear")
        res_y = detrend(res_y, type="linear")
        res_z = detrend(res_z, type="linear")

        res_x -= res_x.mean()
        res_y -= res_y.mean()
        res_z -= res_z.mean()

        acov_x = self.estimate_acov(res_x, max_lag=self.lagNum - 1)
        acov_y = self.estimate_acov(res_y, max_lag=self.lagNum - 1)
        acov_z = self.estimate_acov(res_z, max_lag=self.lagNum - 1)

        # res = detrend(res)
        # res -= np.mean(res)

        # p_max = min(50, n // 10)
        # p = self.pick_AR_order(res, p_max=p_max, ic='bic')
        #
        # phi, sigma2 = self.estimate_ARp(res, p)
        # cov = self.AR_toeplitz_cov(phi, sigma2, n)

        # res = detrend(res, type='linear')
        # acov = self.estimate_acov(res_x, max_lag=4320)
        # acov = self.estimate_acov(res, max_lag=(int(n - 1)))
        # '''FFT'''
        eps_x = max(acov_x[0] * 1e-5, 1e-11)
        #
        # # acov_full = self.psd_clip_autocov(acov, n, eps=eps)
        #
        # # Sigma = toeplitz(acov_full)
        # '''正定化'''
        # # Sigma += np.eye(n) * 1e-2
        # # Sigma = self.nearest_pos_def(Sigma, eps_rel=1e-10)
        #
        # Step 1: DCT-I
        S_x = dct(acov_x, type=1)
        # f = np.linspace(0, 1/(2 * 5), len(S_x))
        # plt.figure(figsize=(7, 4))
        # plt.loglog(f[1:4320], S_x[1:], lw=1.5)
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("PSD (variance)")
        # plt.title("Noise PSD estimated via DCT-I")
        # plt.grid(True)
        # plt.axhline(eps_x, color='r', ls='--', label='eps')
        # plt.tight_layout()
        # plt.show()
        # Step 2: clipping
        S_clipped_x = np.maximum(S_x, eps_x)

        # Step 3: inverse DCT-I
        acov_corr_x = idct(S_clipped_x, type=1, norm=None)
        # plt.plot(np.arange(len(acov_corr)), acov_corr)
        # plt.show()
        Sigma_x = toeplitz(acov_corr_x)

        # '''FFT'''
        eps_y = max(acov_y[0] * 1e-5, 1e-11)
        S_y= dct(acov_y, type=1)
        S_clipped_y = np.maximum(S_y, eps_y)
        acov_corr_y = idct(S_clipped_y, type=1, norm=None)
        Sigma_y = toeplitz(acov_corr_y)

        # '''FFT'''
        eps_z = max(acov_z[0] * 1e-5, 1e-11)
        S_z = dct(acov_z, type=1)
        S_clipped_z = np.maximum(S_z, eps_z)
        acov_corr_z = idct(S_clipped_z, type=1, norm=None)
        Sigma_z = toeplitz(acov_corr_z)

        N = len(acov_x)
        Sigma = np.zeros((3 * N, 3 * N))

        for i in range(N):
            for j in range(N):
                Sigma[3 * i, 3 * j] = Sigma_x[i, j]
                Sigma[3 * i + 1, 3 * j + 1] = Sigma_y[i, j]
                Sigma[3 * i + 2, 3 * j + 2] = Sigma_z[i, j]

        return Sigma


        # return Sigma_x, Sigma_y, Sigma_z

    def getSigma5(self, acov):
        # FFT
        eps = max(acov[0] * 1e-3, 1e-16)
        # Step 1: DCT-I
        S = dct(acov, type=1, norm='ortho')
        f = np.linspace(0, 1 / (2 * 5), len(S))
        plt.figure(figsize=(7, 4))
        plt.loglog(f[1:], S[1:], lw=1.5)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD (variance)")
        plt.title("Noise PSD estimated via DCT-I")
        plt.grid(True)
        plt.axhline(eps, color='r', ls='--', label='eps')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # Step 2: clipping
        S_clipped = np.maximum(S, eps)
        # Step 3: inverse DCT-I
        acov_corr = idct(S_clipped, type=1, norm='ortho')
        Sigma = toeplitz(acov_corr)
        return Sigma

    def getSigma6(self, acov_x, acov_y, acov_z):

        eps_x = max(acov_x[0] * 1e-5, 1e-11)
        #
        # # acov_full = self.psd_clip_autocov(acov, n, eps=eps)
        #
        # # Sigma = toeplitz(acov_full)
        # '''正定化'''
        # # Sigma += np.eye(n) * 1e-2
        # # Sigma = self.nearest_pos_def(Sigma, eps_rel=1e-10)
        #
        # Step 1: DCT-I
        S_x = dct(acov_x, type=1)
        f = np.linspace(0, 1/(2 * 5), len(S_x))
        plt.figure(figsize=(7, 4))
        plt.loglog(f[1:4320], S_x[1:], lw=1.5)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD (variance)")
        plt.title("Noise PSD estimated via DCT-I")
        plt.grid(True)
        plt.axhline(eps_x, color='r', ls='--', label='eps')
        plt.tight_layout()
        plt.show()
        # Step 2: clipping
        S_clipped_x = np.maximum(S_x, eps_x)

        # Step 3: inverse DCT-I
        acov_corr_x = idct(S_clipped_x, type=1, norm=None)
        # plt.plot(np.arange(len(acov_corr)), acov_corr)
        # plt.show()
        Sigma_x = toeplitz(acov_corr_x)

        # '''FFT'''
        eps_y = max(acov_y[0] * 1e-5, 1e-11)
        S_y= dct(acov_y, type=1)
        S_clipped_y = np.maximum(S_y, eps_y)
        acov_corr_y = idct(S_clipped_y, type=1, norm=None)
        Sigma_y = toeplitz(acov_corr_y)

        # '''FFT'''
        eps_z = max(acov_z[0] * 1e-5, 1e-11)
        S_z = dct(acov_z, type=1)
        S_clipped_z = np.maximum(S_z, eps_z)
        acov_corr_z = idct(S_clipped_z, type=1, norm=None)
        Sigma_z = toeplitz(acov_corr_z)

        N = len(acov_x)
        Sigma = np.zeros((3 * N, 3 * N))

        for i in range(N):
            for j in range(N):
                Sigma[3 * i, 3 * j] = Sigma_x[i, j]
                Sigma[3 * i + 1, 3 * j + 1] = Sigma_y[i, j]
                Sigma[3 * i + 2, 3 * j + 2] = Sigma_z[i, j]

        return Sigma


        # return Sigma_x, Sigma_y, Sigma_z

    def getSigma3(self, res):
        res = np.array(res).reshape((-1, 3))
        res_x = res[:, 0]
        res_y = res[:, 1]
        res_z = res[:, 2]

        res_x = detrend(res_x, type="linear")
        res_y = detrend(res_y, type="linear")
        res_z = detrend(res_z, type="linear")

        res_x -= res_x.mean()
        res_y -= res_y.mean()
        res_z -= res_z.mean()

        acov_x = self.estimate_acov(res_x, max_lag=self.lagNum - 1)
        acov_y = self.estimate_acov(res_y, max_lag=self.lagNum - 1)
        acov_z = self.estimate_acov(res_z, max_lag=self.lagNum - 1)

        Sigma_x = self.build_banded_toeplitz(acov_x, self.lagNum, self.lagNum)
        Sigma_y = self.build_banded_toeplitz(acov_y, self.lagNum, self.lagNum)
        Sigma_z = self.build_banded_toeplitz(acov_z, self.lagNum, self.lagNum)

        return Sigma_x, Sigma_y, Sigma_z

    def getSigma4(self, res):
        n = len(res)
        res = detrend(res)
        # res = detrend(res)
        # res -= np.mean(res)

        # p_max = min(50, n // 10)
        # p = self.pick_AR_order(res, p_max=p_max, ic='bic')
        #
        # phi, sigma2 = self.estimate_ARp(res, p)
        # cov = self.AR_toeplitz_cov(phi, sigma2, n)

        # res = detrend(res, type='linear')
        acov = self.estimate_acov(res, max_lag=self.lagNum - 1)

        Sigma = self.build_banded_toeplitz(acov, self.lagNum, 1080)

        return Sigma

    def estimate_acov(self, residuals, max_lag=50):
        """
        用残差估计自协函数（ACF），无偏估计版本。
        """
        residuals = np.asarray(residuals)
        n = len(residuals)

        acov = np.zeros(max_lag + 1)
        for k in range(max_lag + 1):
            acov[k] = np.dot(residuals[:n - k], residuals[k:]) / (n - k)
        return acov
        # n = len(residuals)
        # x = residuals - np.mean(residuals)
        # return np.array([np.dot(x[:n - k], x[k:]) / (n - k) for k in range(max_lag + 1)])

    def nearest_pos_def(self, mat, eps_rel=1e-8):
        # mat = 0.5 * (mat + mat.T)
        # vals, vecs = eigh(mat)
        # max_val = vals[-1] if vals.size > 0 else 0.0
        # tol = eps_rel * max(1.0, max_val)
        # vals_clipped = np.clip(vals, tol, None)
        # mat_pd = (vecs @ np.diag(vals_clipped)) @ vecs.T
        # Sigma_pd = mat + eps_rel * np.eye(mat.shape[0])
        # return (mat_pd + mat_pd.T) / 2.0

        n = mat.shape[0]

        # 只提取最小的 k 个特征值
        vals, vecs = eigsh(mat, k=20, which='SA')

        min_val = np.min(vals)
        if min_val > eps_rel:
            return mat  # 已正定

        # 修正负特征值
        vals_clipped = np.maximum(vals, eps_rel)

        # 只更新部分特征值贡献（低秩修补）
        Sigma_fix = mat + vecs @ np.diag(vals_clipped - vals) @ vecs.T

        return Sigma_fix

    def psd_clip_autocov(self, acov, n_out=None, eps=1e-20):
        """
            Method A: FFT → PSD clipping → IFFT.
            acov: length K+1  (estimated gamma(0..K))
            n_out: desired size of autocov for Toeplitz (same as observation count n)
            eps: lower bound of PSD to enforce strict positive definiteness
            """
        K = len(acov) - 1

        # Step 1: Make FFT input symmetric: g = [gamma0, gamma1,...gammaK, 0,...,0, gammaK,...gamma1]
        # Choose FFT length >= 2*K for good spectral resolution
        m = int(2 ** np.ceil(np.log2(max(2 * (K + 1), 4 * n_out))))
        g = np.zeros(m)
        g[:K + 1] = acov
        g[m - K:m] = acov[1:K+1][::-1]  # symmetry (exclude gamma0 to avoid double counting)

        # Step 2: FFT -> PSD
        S = np.fft.rfft(g)

        S_real = np.real(S)

        # Step 3: PSD clipping (key step)
        S_clipped = np.maximum(S_real, eps)

        # Step 4: inverse FFT
        g_corr = np.fft.irfft(S_clipped).real

        # Step 5: real autocov lags = 0..n_out-1
        acov_corr = g_corr[:n_out]
        return acov_corr

    def taper_keep0_window(self, K, kind='hann'):
        if kind == 'hann':
            w = np.hanning(K + 1)
        elif kind == 'exp':
            k = np.arange(K + 1)
            w = np.exp(-k / (K / 5.0))
        else:
            w = np.ones(K + 1)
        w[0] = 1.0  # 关键：保留 gamma(0)
        return w

    def psd_clip_with_taper(self, acov, n_out,
                            pad_factor=4,
                            eps_rel=1e-10,
                            freq_smooth=11,
                            taper_kind='hann',
                            renormalize=True):
        # acov length = K+1
        K = len(acov) - 1
        if K < 1:
            raise ValueError("acov too short")
        gamma0 = float(acov[0])

        # taper but keep gamma0
        w = self.taper_keep0_window(K, kind=taper_kind)
        acov_tap = acov * w

        # choose FFT length m (power of 2)
        m_candidate = max(2 * (K + 1), pad_factor * n_out)
        m = 1 << int(np.ceil(np.log2(m_candidate)))

        # construct symmetric padded g
        g = np.zeros(m, dtype=float)
        g[:K + 1] = acov_tap
        if K >= 1:
            g[m - K:m] = acov_tap[1:K + 1][::-1]

        # rfft, smooth, clip
        S = np.fft.rfft(g)
        S_real = S.real
        if freq_smooth and freq_smooth > 1:
            S_smooth = uniform_filter1d(S_real, size=freq_smooth, mode='reflect')
        else:
            S_smooth = S_real

        # eps relative to gamma0; if big negative excursions exist, lift scale
        eps_floor = max(eps_rel * max(acov_tap[0], gamma0), 1e-18)
        if S_smooth.min() < 0:
            eps = max(eps_floor, -S_smooth.min() * 1.001)
        else:
            eps = eps_floor

        S_clipped = np.maximum(S_smooth, eps)

        # inverse
        g_corr = np.fft.irfft(S_clipped, n=m).real
        acov_corr = g_corr[:n_out].copy()

        # optional renormalize to preserve gamma0
        if renormalize:
            if acov_corr[0] <= 0:
                raise RuntimeError("acov_corr[0] <= 0 after clipping; increase eps_rel or freq_smooth")
            acov_corr *= (gamma0 / acov_corr[0])

        # diagnostics
        Sigma = toeplitz(acov_corr[:n_out])
        eigs = eigh(Sigma, eigvals_only=True)
        diag = {
            'm': m,
            'gamma0_orig': gamma0,
            'gamma0_tap': acov_tap[0],
            'gamma0_corr': acov_corr[0],
            'S_real_min': float(S_real.min()),
            'S_smooth_min': float(S_smooth.min()),
            'S_clipped_min': float(S_clipped.min()),
            'Sigma_min_eig': float(eigs.min()),
            'Sigma_cond': float(eigs.max() / max(eigs.min(), 1e-30)),
            'eps_used': eps
        }
        return acov_corr, diag

    # def estimate_ARp(self, res, p):
    #     # 去均值
    #     r = res - np.mean(res)
    #     # Yule-Walker 估计 AR(p)
    #     rho, sigma = yule_walker(r, order=p, method='mle')
    #     return rho, sigma

    def AR_toeplitz_cov(self, phi, sigma2, n):
        p = len(phi)
        # 用混合系统求自协
        # Companion matrix
        F = np.zeros((p, p))
        F[0, :] = phi
        F[1:, :-1] = np.eye(p - 1)

        # 计算协方差前 p 项
        gam = np.zeros(n)
        # solve Lyapunov equation for gamma(0:p)
        from numpy.linalg import solve

        # 利用 Yule-Walker 取 gamma(0:p)
        # gamma(0) = sigma^2 / (1 - sum(phi_i * psi_i))（通过 Y-W 已经获得）
        # 用前 p 个自协构造初始向量
        # 我们直接用求解 Lyapunov eq 的方法
        Q = np.zeros((p, p))
        Q[0, 0] = sigma2

        # 解稳定 AR(p) 的协方差矩阵（离散 Lyapunov）

        P = solve_discrete_lyapunov(F, Q)  # p×p

        # 前 p 个自协
        gam[:p] = P[0, :]

        # 后续用递推
        for k in range(p, n):
            gam[k] = np.dot(phi, gam[k - p:k][::-1])

        # 返回 Toeplitz
        return toeplitz(gam[:n])

    # def pick_AR_order(self, res, p_max=50, ic='bic'):
    #     """
    #         自动选择 AR 阶数
    #         """
    #     res = np.asarray(res)
    #     aic_list = []
    #     bic_list = []
    #     orders = list(range(1, p_max + 1))
    #     for p in orders:
    #         try:
    #             model = AutoReg(res, lags=p, old_names=False).fit()
    #             aic_list.append(model.aic)
    #             bic_list.append(model.bic)
    #         except:
    #             aic_list.append(np.inf)
    #             bic_list.append(np.inf)
    #     ic_list = aic_list if ic.lower().startswith('a') else bic_list
    #     p_opt = orders[int(np.argmin(ic_list))]
    #     return p_opt

    def ar_to_acov(self, phi, sigma2, n_lags):
        """
            根据 AR 系数生成自协方差序列
            phi: AR(p) 系数
            sigma2: 白噪声方差
            n_lags: 生成多少滞后
            """
        p = len(phi)
        acov = np.zeros(n_lags)
        acov[0] = sigma2 / (1 - np.sum(phi)) ** 2  # 零滞后
        for k in range(1, n_lags):
            # 用 AR 递推生成 acov[k]（简单近似）
            acov[k] = np.dot(phi[:min(k, p)], acov[k - 1:k - p - 1:-1])
        return acov

    def build_banded_toeplitz(self, acov, N, K, eps_factor=1e-8):
        """
        acov       : 自协函数 γ(0..)
        N          : 弧段长度（如 4320）
        K          : 最大滞后（带宽）
        eps_factor : nugget 强度
        """
        assert K < N
        assert len(acov) >= K + 1

        Sigma = np.zeros((N, N))

        # 主对角 + K 条副对角
        for k in range(K + 1):
            val = acov[k]
            if k == 0:
                Sigma += np.diag(val * np.ones(N))
            else:
                Sigma += np.diag(val * np.ones(N - k), k)
                Sigma += np.diag(val * np.ones(N - k), -k)

        # 强制对称（数值安全）
        Sigma = 0.5 * (Sigma + Sigma.T)

        # 对角加载（nugget）
        eps = eps_factor * acov[0]
        Sigma += eps * np.eye(N)

        return Sigma

    def saveSigma1(self, Sigma, h5path):

        if "Sigma" in h5path:
            del h5path["Sigma"]
        #
        # if "Sigma_y" in h5path:
        #     del h5path["Sigma_y"]
        # #
        # if "Sigma_z" in h5path:
        #     del h5path["Sigma_z"]
        #
        # h5path.create_dataset("Sigma_x", data=Sigma_x)
        # h5path.create_dataset("Sigma_y", data=Sigma_y)
        h5path.create_dataset("Sigma", data=Sigma)
        h5path.close()
        return self

    def saveSigma2(self, Sigma, h5path):

        if "Sigma" in h5path:
            del h5path["Sigma"]

        h5path.create_dataset("Sigma", data=Sigma)
        h5path.close()
        return self