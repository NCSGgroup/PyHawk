import os.path
import sys
sys.path.append("../")

from src.Interface.ArcOperation import ArcData
import src.Preference.EnumType as EnumType
import numpy as np
from tqdm import tqdm
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.Solver.AdjustOrbit import AdjustOrbit
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import pathlib
import h5py
import multiprocessing
from src.Solver.DesignMatrix import GravityDesignMat
from src.Solver.NEQ import GravityNEQperArc, GravityNEQperArc_v2, GravityNEQperArc_v3
from src.Auxilary.SortCS import SortCS
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Auxilary.StandardNEQ import StandardNEQ
from src.Auxilary.StandardACC import StandardACC
from src.Auxilary.StandardKBRR import StandardKBRR
from src.Auxilary.StandardLGD import StandardLGD
from src.Interface.LoadSH import LoadGif48,LoadGOCO06S
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
from src.Auxilary.StandardSH import StandardSH
from src.Auxilary.StandardConfig import StandardConfig
import multiprocessing
import json
from datetime import datetime
from src.Auxilary.Outlier import Outlier, ArcOutlier
from src.Preference.EnumType import Payload, SatID, SSTObserve, StaticGravModel
from scipy.signal import butter, filtfilt
from scipy import signal
from scipy.linalg import toeplitz, cholesky, eigh, solve_triangular
from scipy.sparse.linalg import eigsh
from scipy.signal import detrend, welch
from scipy.ndimage import uniform_filter1d
from scipy.fft import dct, idct
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import block_diag
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from numpy.fft import rfft, irfft
from src.Solver.LGD import ComputeLGD
from src.Solver.LGDFromKBRA import ComputeLGDFromKBRA
from src.Solver.CovarianceModel import (
    load_covariance,
    orbit_covariance_filename,
    range_rate_covariance_filename,
    save_stationary_covariance,
)


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
        self.__lgd_config = None
        self.__standard_config = None
        self.__arc_path = None
        self.__arcft_path = None
        self.__kbrr_qc_records = []
        self._gls_rng = np.random.default_rng(0)
        pass

    def loadJson(self, setting_directory=None):
        if setting_directory is None:
            setting_directory = os.path.join(
                self.__parent_path, 'setting', 'Calibrate'
            )
        setting_directory = os.path.abspath(
            os.fspath(setting_directory)
        )

        def load_setting(filename):
            with open(
                    os.path.join(setting_directory, filename),
                    'r', encoding='utf-8') as file:
                return json.load(file)

        self.__acc_config_A = AccelerometerConfig()
        self.__acc_config_A.__dict__ = load_setting(
            'AccelerometerConfig_A.json'
        )

        self.__acc_config_B = AccelerometerConfig()
        self.__acc_config_B.__dict__ = load_setting(
            'AccelerometerConfig_B.json'
        )

        self.__design_acc_config_A = AccelerometerConfig()
        self.__design_acc_config_A.__dict__ = load_setting(
            'DesignAccelerometerConfig_A.json'
        )

        self.__design_acc_config_B = AccelerometerConfig()
        self.__design_acc_config_B.__dict__ = load_setting(
            'DesignAccelerometerConfig_B.json'
        )

        self.__force_model_config = ForceModelConfig()
        self.__force_model_config.__dict__ = load_setting(
            'ForceModelConfig.json'
        )

        self.__parameter_config = ParameterConfig()
        self.__parameter_config.__dict__ = load_setting(
            'OrbitFitParameterConfig.json'
        )

        self.__solver_config = SolverConfig()
        self.__solver_config.__dict__ = load_setting('SolverConfig.json')

        self.__ode_config = ODEConfig()
        self.__ode_config.__dict__ = load_setting('ODEConfig.json')

        self.__neq_config = NEQConfig()
        self.__neq_config.__dict__ = load_setting('NEQConfig.json')

        self.__adjust_config = AdjustOrbitConfig()
        self.__adjust_config.__dict__ = load_setting(
            'AdjustOrbitConfig.json'
        )

        self.__interface_config = InterfaceConfig()
        self.__interface_config.__dict__ = load_setting(
            'InterfaceConfig.json'
        )

        self.__design_parameter_config = ParameterConfig()
        self.__design_parameter_config.__dict__ = load_setting(
            'GravityDesignParameterConfig.json'
        )

        self.__frame_config = FrameConfig()
        self.__frame_config.__dict__ = load_setting('FrameConfig.json')

        self.__calibrateOrbit_config = CalibrateOrbitConfig()
        self.__calibrateOrbit_config.__dict__ = load_setting(
            'CalibrateOrbitConfig.json'
        )

        neq_path_config = self.__neq_config.PathOfFiles()
        neq_path_config.__dict__.update(
            self.__neq_config.PathOfFilesConfig.copy()
        )
        stokes_config = self.__design_parameter_config.StokesCoefficients()
        stokes_config.__dict__.update(
            self.__design_parameter_config.StokesCoefficientsConfig.copy()
        )
        expected_sh_filename = StandardSH.product_filename(
            data=self.__interface_config.date_span,
            degree=stokes_config.MaxDegree,
            mission=self.__interface_config.mission
        )
        expected_sh_path = pathlib.Path(neq_path_config.ResultCS).joinpath(
            pathlib.Path(expected_sh_filename).stem,
            expected_sh_filename
        )
        self.__standard_config = StandardConfig(setting_directory)
        self.__standard_config.save(expected_sh_path)
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
        self.__arc_path = arc_path
        self.__arcft_path = arcft_path
        self.__arcLen = None

        '''get step and process control'''
        calibrateOrbit_config = self.__calibrateOrbit_config
        MissionConfig = CalibrateOrbitConfig.Mission()
        MissionConfig.__dict__.update(calibrateOrbit_config.MissionConfig.copy())
        StepControl = CalibrateOrbitConfig.StepControl()
        StepControl.__dict__.update(calibrateOrbit_config.StepControlConfig.copy())
        ParallelControl = CalibrateOrbitConfig.ParallelControl()
        ParallelControl.__dict__.update(calibrateOrbit_config.ParallelControlConfig.copy())
        GLSIteration = CalibrateOrbitConfig.GLSIteration()
        GLSIteration.__dict__.update(
            getattr(calibrateOrbit_config, 'GLSIterationConfig', {}).copy()
        )
        LGDConfig = CalibrateOrbitConfig.LGD()
        LGDConfig.__dict__.update(
            getattr(calibrateOrbit_config, 'LGDConfig', {}).copy()
        )
        self.__lgd_config = LGDConfig
        isPreprocess = StepControl.isPreprocess
        isAdjustOrbit = StepControl.isAdjustOrbit
        isAdjustKBRR = StepControl.isAdjustKBRR
        isGetLGD = StepControl.isGetLGD
        isGravityDesignMat = StepControl.isGetGravityDesignMat
        isNEQ = StepControl.isGetNEQ
        isSH = StepControl.isGetSH
        isGLS = StepControl.isGLS

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
            standard_acc_writers = []
            standard_acc_paths = set()
            for acc_config in (self.__acc_config_A, self.__acc_config_B):
                path = os.path.abspath(
                    getattr(
                        acc_config,
                        'StandardACCTemp',
                        getattr(
                            acc_config,
                            'temp_standard_acc_data',
                            StandardACC.output_directory(
                                acc_config.temp_non_conservative_data
                            )
                        )
                    )
                )
                if path in standard_acc_paths:
                    continue
                standard_acc_paths.add(path)
                writer = StandardACC().configure(
                    filedir=path,
                    date_span=self.__date_span,
                    mission=MissionConfig.mission
                )
                writer.prepare(
                    satellites=[key for key, value in self.__sat.items() if value]
                )
                standard_acc_writers.append(writer)
            pool = multiprocessing.Pool(processes=AdjustOrbitProcess)
            # for i in np.arange(0, 10, 1):
            #     self._adjustOrbit(i)
            pool.map(self._adjustOrbit, self.__arclist)
            pool.close()
            pool.join()
            for writer in standard_acc_writers:
                writer.finalize()
        self.__saveOrbitRMS()
        standard_pre_kbrr = None
        kbrr_input_arcs = list(self.__arclist)
        if isAdjustKBRR:
            self.makeLog(stepName='adjustKBRR', process=AdjustKBRRProcess)
            adjust_path = self.__adjust_config.PathOfFiles()
            adjust_path.__dict__.update(
                self.__adjust_config.PathOfFilesConfig.copy()
            )
            standard_kbrr_directory = getattr(
                adjust_path,
                'StandardKBRRTemp',
                StandardKBRR.output_directory(adjust_path.RangeRateTemp)
            )
            standard_pre_kbrr = StandardKBRR().configure(
                filedir=standard_kbrr_directory,
                date_span=self.__date_span,
                stage='Pre',
                mission=MissionConfig.mission
            ).prepare()
            pool = multiprocessing.Pool(processes=AdjustKBRRProcess)
            pool.map(self._adjustKBRR, self.__arclist)
            pool.close()
            pool.join()
        self.__arclist, rejected_kbrr_arcs = self.__screenKBRRArcs(
            kbrr_input_arcs
        )
        if standard_pre_kbrr is not None and rejected_kbrr_arcs:
            standard_pre_kbrr.discard_arcs(rejected_kbrr_arcs)
        self.__saveKBRRRMS(arc_numbers=kbrr_input_arcs)
        if isGetLGD:
            self.makeLog(stepName='LGD', process=AdjustKBRRProcess)
            adjust_path = self.__adjust_config.PathOfFiles()
            adjust_path.__dict__.update(
                self.__adjust_config.PathOfFilesConfig.copy()
            )
            standard_lgd = StandardLGD().configure(
                filedir=getattr(
                    adjust_path,
                    'StandardLGDTemp',
                    '../temp/StandardLGD'
                ),
                date_span=self.__date_span,
                mission=MissionConfig.mission
            ).prepare()
            pool = multiprocessing.Pool(
                processes=AdjustKBRRProcess
            )
            pool.map(
                self._computeLGD,
                self.__arclist
            )
            pool.close()
            pool.join()
            standard_lgd.finalize()
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
            pool.map(self._NEQ, [(q, False) for q in self.__arclist])
            # pool.map(self._NEQ, (0, 1))
            pool.close()
            pool.join()
        if isSH:
            self.makeLog(stepName='SH', process=1)
            self.__calculate_cs(int(self.__solver_config.OrbitKinFactor))
            self.plotPostKbrr()
            # self.xyz2lonlat()
        if isGLS:
            self._iterative_gls(
                process_count=GetNEQProcess,
                iteration_config=GLSIteration
            )
        if standard_pre_kbrr is not None:
            standard_pre_kbrr.cleanup()
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

    def _computeLGD(self, arcNo):
        lgd_config = self.__lgd_config or CalibrateOrbitConfig.LGD()
        method = getattr(lgd_config, 'Method', 'RangeRateDerivative')
        method_key = str(method).replace('_', '').replace('-', '').lower()

        if method_key in ('rangeratederivative', 'kbrrderivative', 'numericaldifferentiation'):
            ComputeLGD(
                arcNo=arcNo,
                date_span=self.__date_span
            ).configure(
                AdjustOrbitConfig=self.__adjust_config,
                InterfaceConfig=self.__interface_config,
                FrameConfig=self.__frame_config,
                f_low=getattr(lgd_config, 'FLowHz', 1.0e-3),
                f_high=getattr(lgd_config, 'FHighHz', None),
                filter_order=getattr(lgd_config, 'FilterOrder', 6),
                gap_threshold=getattr(lgd_config, 'GapThreshold', 20.0),
                min_segment_length=getattr(lgd_config, 'MinSegmentLength', 300),
                kbrr_preprocess_enable=getattr(lgd_config, 'KBRRPreprocessEnable', True),
                kbrr_despike_enable=getattr(lgd_config, 'KBRRDespikeEnable', True),
                kbrr_despike_window=getattr(lgd_config, 'KBRRDespikeWindow', 121),
                kbrr_despike_sigma=getattr(lgd_config, 'KBRRDespikeSigma', 6.0),
                kbrr_remove_low_frequency=getattr(lgd_config, 'KBRRRemoveLowFrequency', True),
                kbrr_trend_degree=getattr(lgd_config, 'KBRRTrendDegree', 2),
                kbrr_cpr_period=getattr(lgd_config, 'KBRRCPRPeriod', 5400.0),
                kbrr_cpr_harmonics=getattr(lgd_config, 'KBRRCPRHarmonics', 2),
                segment_edge_trim_seconds=getattr(lgd_config, 'SegmentEdgeTrimSeconds', 60.0)
            ).compute()
        elif method_key in ('referencekbra', 'kbrareference', 'dynamickbra'):
            ComputeLGDFromKBRA(
                arcNo=arcNo,
                date_span=self.__date_span
            ).configure(
                AdjustOrbitConfig=self.__adjust_config,
                InterfaceConfig=self.__interface_config,
                FrameConfig=self.__frame_config,
                payload=getattr(lgd_config, 'ReferenceKBRAPayload', 'KBR'),
                allow_velocity_fallback=getattr(
                    lgd_config,
                    'ReferenceKBRAAllowVelocityFallback',
                    False
                )
            ).compute_from_arc(
                output_name='LGD_' + str(arcNo) + '.hdf5'
            )
        else:
            raise ValueError(
                "Unsupported LGDConfig.Method: %s. Use 'RangeRateDerivative' or 'ReferenceKBRA'." % method
            )

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

    def _NEQ(self, args):
        '''setp 8: get NEQ'''
        q, use_sigma = args
        neq = GravityNEQperArc(arcNo=q, date_span=self.__date_span, sat=self.__sat)
        neq.configure(solverConfig=self.__solver_config, NEQConfig=self.__neq_config,
                      AdjustOrbitConfig=self.__adjust_config, InterfaceConfig=self.__interface_config).get_neq(use_sigma=use_sigma)
        pass

    def _build_neq(self, use_sigma, process_count):
        args = [(arc, use_sigma) for arc in self.__arclist]
        if process_count == 1:
            for arg in args:
                self._NEQ(arg)
            return
        # h5py is not fork-safe when the parent has opened HDF5 files.
        # Spawn clean workers so each process opens its own files.
        context = multiprocessing.get_context('spawn')
        with context.Pool(processes=process_count) as pool:
            pool.map(self._NEQ, args)

    def _iterative_gls(self, process_count, iteration_config):
        max_iteration = max(1, int(iteration_config.MaxIteration))
        min_iteration = max(1, int(iteration_config.MinIteration))
        relative_tolerance = float(iteration_config.RelativeTolerance)
        absolute_tolerance = float(iteration_config.AbsoluteTolerance)
        orbit_covariance_mode = self._normalize_orbit_covariance_mode(
            getattr(
                iteration_config, 'OrbitCovarianceMode', 'FixedDiagonal'
            )
        )
        orbit_fixed_diagonal = None
        if orbit_covariance_mode == 'FixedDiagonal':
            orbit_fixed_diagonal = float(
                getattr(iteration_config, 'OrbitFixedDiagonal', 0.004)
            )
            if (not np.isfinite(orbit_fixed_diagonal) or
                    orbit_fixed_diagonal <= 0.0):
                raise ValueError(
                    'OrbitFixedDiagonal must be finite and positive'
                )
        previous_cs = None
        random_seed = int(
            getattr(iteration_config, 'RandomSeed', 20240724)
        )

        for iteration in range(max_iteration):
            iteration_no = iteration + 1
            print('GLS iteration {}/{}'.format(iteration_no, max_iteration))
            # Use a deterministic but iteration-specific stream so repeated
            # runs are reproducible without filling every iteration identically.
            self._gls_rng = np.random.default_rng(random_seed + iteration)

            # Build a stochastic model from the posterior residuals of the
            # previous solution, then rebuild the weighted normal equations.
            if orbit_covariance_mode == 'FixedDiagonal':
                self.GLS(orbit_fixed_diagonal=orbit_fixed_diagonal)
            else:
                self.GLS_full()
            self._build_neq(use_sigma=True, process_count=process_count)
            current_cs = self.__calculate_cs(OrbitKinFactor=1, use_sigma=True)

            gravity_start = 18 if self.AccelerometerConfig.isScale else 0
            current_gravity = current_cs[gravity_start:]

            if previous_cs is not None:
                absolute_change = np.linalg.norm(current_gravity - previous_cs)
                scale = max(
                    np.linalg.norm(current_gravity),
                    np.linalg.norm(previous_cs),
                    np.finfo(float).eps
                )
                relative_change = absolute_change / scale
                print(
                    'GLS iteration {}: absolute change={:.6e}, '
                    'relative change={:.6e}'.format(
                        iteration_no, absolute_change, relative_change
                    )
                )

                if iteration_no >= min_iteration and (
                        relative_change <= relative_tolerance or
                        absolute_change <= absolute_tolerance):
                    print('GLS converged after {} iterations'.format(iteration_no))
                    break

            previous_cs = current_gravity.copy()
        else:
            print('GLS reached the maximum of {} iterations'.format(max_iteration))

    @staticmethod
    def _normalize_orbit_covariance_mode(mode):
        aliases = {
            'fixeddiagonal': 'FixedDiagonal',
            'diagonal': 'FixedDiagonal',
            'estimatedfull': 'EstimatedFull',
            'full': 'EstimatedFull'
        }
        key = str(mode).replace('_', '').replace('-', '').lower()
        if key not in aliases:
            raise ValueError(
                'Unsupported OrbitCovarianceMode {!r}; expected '
                'FixedDiagonal or EstimatedFull'.format(mode)
            )
        return aliases[key]

    def _NEQ_v3(self, q):
        '''setp 8: get NEQ'''
        neq = GravityNEQperArc_v3(arcNo=q, date_span=self.__date_span, sat=self.__sat)
        neq.configure(solverConfig=self.__solver_config, NEQConfig=self.__neq_config,
                      AdjustOrbitConfig=self.__adjust_config, InterfaceConfig=self.__interface_config).get_neq()
        pass

    def __calculate_cs(self, OrbitKinFactor, use_sigma=False):
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
        '''config force model'''
        RefGravModelConfig = self.__force_model_config.RefGravModel()
        RefGravModelConfig.__dict__.update(self.__force_model_config.RefGravModelConfig.copy())
        staticModel = RefGravModelConfig.StaticModel
        time_variable_date = getattr(RefGravModelConfig, 'TimeVariableDate', None)
        static_model_file = None
        static_model_epoch = None

        if staticModel == StaticGravModel.Gif48.name:
            static_model_file = force_model_path_config.Gif48
            self.__staticModel = LoadGif48().load(static_model_file)
        elif staticModel == StaticGravModel.EIGEN6_C4.name:
            static_model_file = force_model_path_config.EIGEN6_C4
            self.__staticModel = LoadGif48().load(static_model_file)
        elif staticModel == StaticGravModel.GOCO02s.name:
            static_model_file = force_model_path_config.GOCO02s
            self.__staticModel = LoadGif48().load(static_model_file)
        elif staticModel == StaticGravModel.GGM05C.name:
            static_model_file = force_model_path_config.GGM05C
            self.__staticModel = LoadGif48().load(static_model_file)
        elif staticModel == StaticGravModel.GOCO06S.name:
            static_model_file = force_model_path_config.GOCO06s
            static_model_epoch = time_variable_date
            self.__staticModel = LoadGOCO06S().load(
                static_model_file,
                date=static_model_epoch
            )

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
            try:
                RangeRate_N = h51['RangeRate_N'][:]
            except KeyError:
                print(" Missing dataset: RangeRate_N skip")
                continue

            try:
                Orbit_N_A = h51['Orbit_N_A'][:]
            except KeyError:
                print(" Missing dataset: RangeRate_N skip")
                continue

            try:
                Orbit_N_B = h51['Orbit_N_B'][:]
            except KeyError:
                print(" Missing dataset: RangeRate_N skip")
                continue

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
        savefile = res_dir1 + '/' + date_span[0] + '-' + date_span[1] + '-NormalEquation.hdf5'
        with h5py.File(savefile, 'w') as h5:

            h5.create_dataset(
                'NMatrix',
                data=NMatrix1
            )

            h5.create_dataset(
                'LMatrix',
                data=LMatrix1
            )

            h5.attrs['parameter_number'] = parameter_number
            h5.attrs['OrbitKinFactor'] = OrbitKinFactor

        scale_parameter_count = 18 if self.AccelerometerConfig.isScale else 0
        try:
            standard_neq_directory = getattr(
                neq_path_config,
                'StandardNEQTemp',
                '../temp/StandardNEQ'
            )
            StandardNEQ().configure(
                filedir=standard_neq_directory,
                date_span=date_span,
                degree_max=stokes_coefficients_config.MaxDegree,
                degree_min=stokes_coefficients_config.MinDegree,
                sort_method=stokes_coefficients_config.SortMethod,
                scale_parameter_count=scale_parameter_count,
                apriori_model=self.__staticModel
            ).save(NMatrix=NMatrix1, LMatrix=LMatrix1)
        except Exception as e:
            print('Save standard NEQ failed: %s' % e)

        cs = np.linalg.lstsq(NMatrix1, LMatrix1[:, 0], rcond=None)[0]
        self.cs=cs
        # cs2 = np.linalg.lstsq(NMatrix2, LMatrix2[:, 0], rcond=None)[0]
        if scale_parameter_count:
            scale = cs[:scale_parameter_count]
            print(scale)
            cs = cs[scale_parameter_count:]

        adjust_path = self.__adjust_config.PathOfFiles()
        adjust_path.__dict__.update(
            self.__adjust_config.PathOfFilesConfig.copy()
        )
        standard_kbrr_directory = getattr(
            adjust_path,
            'StandardKBRRTemp',
            StandardKBRR.output_directory(adjust_path.RangeRateTemp)
        )
        standard_post_kbrr = StandardKBRR().configure(
            filedir=standard_kbrr_directory,
            date_span=self.__date_span,
            stage='Post',
            mission=self.__interface_config.mission
        ).prepare()
        sigma = self.getLocalPar(
            solverConfig=self.__solver_config,
            N=NMatrix1,
            use_sigma=use_sigma
        )
        standard_post_kbrr.finalize()
        if scale_parameter_count:
            sigma = sigma[scale_parameter_count:]
        # sigma = cs

        gravity_parameter_count = (
            (stokes_coefficients_config.MaxDegree + 1)
            * (stokes_coefficients_config.MaxDegree + 2)
            - stokes_coefficients_config.MinDegree
            * (stokes_coefficients_config.MinDegree + 1)
            - (stokes_coefficients_config.MaxDegree
               - stokes_coefficients_config.MinDegree + 1)
        )
        if cs.size != gravity_parameter_count:
            raise ValueError(
                'Solved SH parameter count is %d; configuration requires %d'
                % (cs.size, gravity_parameter_count)
            )
        if sigma.size != gravity_parameter_count:
            raise ValueError(
                'SH uncertainty count is %d; configuration requires %d'
                % (sigma.size, gravity_parameter_count)
            )

        sort = SortCS(method=stokes_coefficients_config.SortMethod,
                      degree_max=stokes_coefficients_config.MaxDegree,
                      degree_min=stokes_coefficients_config.MinDegree)
        c, s = sort.invert(cs)
        sigmaC , sigmaS = sort.invert(sigma)

        static_c, static_s = self.__staticModel.getCS(stokes_coefficients_config.MaxDegree)

        C, S = c + static_c, s + static_s

        standard_sh_path = StandardSH().configure(
            filedir=neq_path_config.ResultCS,
            data=date_span,
            degree=stokes_coefficients_config.MaxDegree,
            c=C,
            s=S,
            sigmaC=sigmaC,
            sigmaS=sigmaS,
            mission=self.__interface_config.mission,
            reference_model=self.__staticModel,
            unused_days=getattr(self.__interface_config, 'date_delete', [])
        ).unfiltered()
        self.__standard_config.archive_quality_control(
            spherical_harmonic_path=standard_sh_path,
            arc_filter_path=self.__arcft_path,
            arc_definition_path=self.__arc_path
        )

        return self.cs.copy()

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
            with h5py.File(res_filename, 'r') as h5:
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

    def __screenKBRRArcs(self, arc_numbers):
        """Reject statistically high KBRR arcs relative to the current month."""
        range_rate_config = self.__adjust_config.RangeRate()
        range_rate_config.__dict__.update(
            self.__adjust_config.RangeRateConfig.copy()
        )
        enabled = bool(getattr(
            range_rate_config, 'ArcOutlierEnabled', True
        ))
        significance = float(getattr(
            range_rate_config, 'ArcOutlierSignificance', 0.01
        ))
        max_fraction = float(getattr(
            range_rate_config, 'ArcOutlierMaxFraction', 0.1
        ))
        min_arcs = int(getattr(
            range_rate_config, 'ArcOutlierMinArcs', 8
        ))

        adjust_path = AdjustOrbitConfig.PathOfFiles()
        adjust_path.__dict__.update(
            self.__adjust_config.PathOfFilesConfig.copy()
        )
        res_dir = pathlib.Path(adjust_path.RangeRateTemp).joinpath(
            self.__date_span[0] + '_' + self.__date_span[1]
        )

        records = []
        metric_record_indices = []
        metrics = []
        for arc_number in arc_numbers:
            path = res_dir.joinpath(
                'RangeRate_' + str(arc_number) + '.hdf5'
            )
            record = {
                'arc': int(arc_number),
                'accepted': False,
                'reason': 'missing_file',
                'pre_count': 0,
                'post_count': 0,
                'completion_ratio': None,
                'rms': None,
                'robust_scale': None,
                'quality_metric': None,
                'esd_order': None,
                'esd_statistic': None,
                'esd_critical_value': None,
                'path': str(path)
            }
            records.append(record)
            if not path.exists():
                continue

            with h5py.File(path, 'r') as h5:
                pre_count = (
                    int(h5['pre_t'].shape[0]) if 'pre_t' in h5 else 0
                )
                residual = (
                    np.asarray(h5['post_residual'][:], dtype=np.float64)
                    if 'post_residual' in h5 else np.zeros(0)
                )
            record['pre_count'] = pre_count
            record['post_count'] = int(residual.size)
            if pre_count:
                record['completion_ratio'] = (
                    float(residual.size) / pre_count
                )
            if residual.size == 0:
                record['reason'] = 'empty_post_residual'
                continue
            if not np.all(np.isfinite(residual)):
                record['reason'] = 'nonfinite_post_residual'
                continue

            arc_rms = float(
                np.linalg.norm(residual) / np.sqrt(residual.size)
            )
            robust_scale = float(
                1.4826 * np.median(np.abs(residual))
            )
            record['rms'] = arc_rms
            record['robust_scale'] = robust_scale
            record['accepted'] = True
            record['reason'] = 'accepted'

            if arc_rms > 0 and robust_scale > 0:
                quality_metric = float(
                    np.sqrt(arc_rms * robust_scale)
                )
                record['quality_metric'] = quality_metric
                metric_record_indices.append(len(records) - 1)
                metrics.append(quality_metric)

        diagnostics = {
            'outlier_count': 0,
            'candidate_indices': np.zeros(0, dtype=np.int64),
            'statistics': np.zeros(0, dtype=np.float64),
            'critical_values': np.zeros(0, dtype=np.float64)
        }
        if enabled and len(metrics) >= min_arcs:
            rejected, diagnostics = ArcOutlier(
                significance=significance,
                max_fraction=max_fraction,
                min_samples=min_arcs
            ).detect(metrics)
            for metric_index, is_rejected in enumerate(rejected):
                if is_rejected:
                    record = records[
                        metric_record_indices[metric_index]
                    ]
                    record['accepted'] = False
                    record['reason'] = 'monthly_upper_esd'

            for order, metric_index in enumerate(
                    diagnostics['candidate_indices']):
                record = records[
                    metric_record_indices[int(metric_index)]
                ]
                record['esd_order'] = order + 1
                record['esd_statistic'] = float(
                    diagnostics['statistics'][order]
                )
                record['esd_critical_value'] = float(
                    diagnostics['critical_values'][order]
                )

        accepted_arcs = [
            record['arc'] for record in records if record['accepted']
        ]
        rejected_arcs = [
            record['arc'] for record in records if not record['accepted']
        ]
        self.__kbrr_qc_records = records
        self.__writeKBRRArcQuality(
            res_dir=res_dir,
            records=records,
            enabled=enabled,
            significance=significance,
            max_fraction=max_fraction,
            min_arcs=min_arcs,
            esd_outlier_count=int(diagnostics['outlier_count'])
        )
        print(
            'KBRR arc quality control retained %d arcs and rejected %d: %s'
            % (
                len(accepted_arcs), len(rejected_arcs),
                ', '.join(str(arc) for arc in rejected_arcs)
            )
        )
        return accepted_arcs, rejected_arcs

    def __writeKBRRArcQuality(self, res_dir, records, enabled,
                              significance, max_fraction, min_arcs,
                              esd_outlier_count):
        """Persist arc decisions without deleting the residual datasets."""
        os.makedirs(res_dir, exist_ok=True)
        method = (
            'one-sided generalized ESD on log(sqrt(RMS * robust_scale))'
        )
        for record in records:
            path = pathlib.Path(record['path'])
            if not path.exists():
                continue
            with h5py.File(path, 'r+') as h5:
                h5.attrs['arc_qc_accepted'] = bool(record['accepted'])
                h5.attrs['arc_qc_reason'] = record['reason']
                h5.attrs['arc_qc_method'] = method
                h5.attrs['arc_qc_rms'] = (
                    np.nan if record['rms'] is None else record['rms']
                )
                h5.attrs['arc_qc_robust_scale'] = (
                    np.nan if record['robust_scale'] is None
                    else record['robust_scale']
                )
                h5.attrs['arc_qc_quality_metric'] = (
                    np.nan if record['quality_metric'] is None
                    else record['quality_metric']
                )
                h5.attrs['arc_qc_esd_statistic'] = (
                    np.nan if record['esd_statistic'] is None
                    else record['esd_statistic']
                )
                h5.attrs['arc_qc_esd_critical_value'] = (
                    np.nan if record['esd_critical_value'] is None
                    else record['esd_critical_value']
                )

        summary = {
            'date_span': list(self.__date_span),
            'created_utc': datetime.utcnow().strftime(
                '%Y-%m-%dT%H:%M:%SZ'
            ),
            'method': method,
            'enabled': enabled,
            'significance': significance,
            'max_outlier_fraction': max_fraction,
            'minimum_monthly_arcs': min_arcs,
            'esd_outlier_count': esd_outlier_count,
            'accepted_arcs': [
                record['arc'] for record in records
                if record['accepted']
            ],
            'rejected_arcs': [
                record['arc'] for record in records
                if not record['accepted']
            ],
            'records': records
        }
        target = res_dir.joinpath('KBRR_ArcQuality.json')
        temporary = pathlib.Path(str(target) + '.tmp')
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2, allow_nan=False)
        os.replace(temporary, target)

    def __saveKBRRRMS(self, arc_numbers=None):
        self.AdjustPath = AdjustOrbitConfig.PathOfFiles()
        self.AdjustPath.__dict__.update(self.__adjust_config.PathOfFilesConfig.copy())

        res_dir = pathlib.Path().joinpath(self.AdjustPath.RangeRateTemp, self.__date_span[0] + '_' + self.__date_span[1])
        if arc_numbers is None:
            arc_numbers = self.__arclist
        plot_arcs = []
        post_RMS = []
        for i in arc_numbers:
            plot_arcs.append(i)
            rms = np.nan
            res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
            if os.path.exists(res_filename):
                with h5py.File(res_filename, 'r') as h5:
                    if 'post_residual' in h5:
                        postfit = h5['post_residual'][()]
                        if postfit.size > 0 and np.all(np.isfinite(postfit)):
                            rms = self.__rms(postfit)
            post_RMS.append(rms)

        arc_numbers = np.asarray(plot_arcs)
        post_RMS = np.asarray(post_RMS) * 1e6
        invalid = ~np.isfinite(post_RMS)
        rejected_arc_numbers = {
            record['arc'] for record in self.__kbrr_qc_records
            if (not record['accepted']
                and record['reason'] == 'monthly_upper_esd')
        }
        rejected = (
            np.isin(arc_numbers, list(rejected_arc_numbers))
            & ~invalid
        )
        plt.figure(dpi=300, figsize=(16, 8))
        plt.title('KBRR-RMS-{}-{}'.format(self.__date_span[0], self.__date_span[-1]), fontsize=24)
        plt.plot(arc_numbers, post_RMS, marker='o', label="arc RMS")
        if np.any(rejected):
            plt.scatter(
                arc_numbers[rejected], post_RMS[rejected],
                marker='x', s=100, linewidths=2, color='red',
                label='rejected arc'
            )
        if np.any(invalid):
            plt.scatter(arc_numbers[invalid], np.zeros(np.count_nonzero(invalid)),
                        marker='x', color='black',
                        label='empty/missing arc')
        plt.yticks(fontsize=18)
        plt.xlabel('Arc number', fontsize=20)
        plt.ylabel(r'Residual RMS [$\mathrm{\mu m/s}$]', fontsize=20)
        plt.legend(loc='upper right', fontsize=20)
        plt.grid(ls='--')
        plt.tight_layout()
        os.makedirs('../result/img', exist_ok=True)
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

    @staticmethod
    def _solve_local_parameters(global_dm, local_dm, obs, global_parameters,
                                covariance=None):
        global_dm = np.asarray(global_dm)
        local_dm = np.asarray(local_dm)
        obs = np.asarray(obs).reshape(-1)

        if global_dm.ndim != 2 or local_dm.ndim != 2:
            raise ValueError(
                'Global and local design matrices must both be two-dimensional'
            )
        if global_dm.shape[0] != obs.size or local_dm.shape[0] != obs.size:
            raise ValueError(
                'Local-parameter row mismatch: global_dm={}, local_dm={}, '
                'obs={}'.format(global_dm.shape, local_dm.shape, obs.shape)
            )

        # scipy.linalg.solve_triangular cannot accept a 0 x 0 triangular
        # system. Empty post-fit blocks can occur after arc screening.
        if obs.size == 0:
            return np.zeros(local_dm.shape[1]), np.empty(0)

        reduced_obs = obs - global_dm @ global_parameters

        if local_dm.shape[1] == 0:
            return np.empty(0), reduced_obs

        if covariance is None:
            local_parameters = np.linalg.lstsq(
                local_dm, reduced_obs, rcond=None
            )[0]
        else:
            covariance = np.asarray(covariance)
            expected_shape = (obs.size, obs.size)
            if covariance.shape != expected_shape:
                raise ValueError(
                    'Local-parameter covariance mismatch: expected {}, got '
                    '{}'.format(expected_shape, covariance.shape)
                )
            covariance = 0.5 * (covariance + covariance.T)
            lower = cholesky(covariance, lower=True)
            local_dm_white = solve_triangular(
                lower, local_dm, lower=True
            )
            reduced_obs_white = solve_triangular(
                lower, reduced_obs, lower=True
            )
            local_parameters = np.linalg.lstsq(
                local_dm_white, reduced_obs_white, rcond=None
            )[0]

        residual = reduced_obs - local_dm @ local_parameters
        return local_parameters, residual

    @staticmethod
    def _load_arc_covariance(filename, target_time, components=1):
        return load_covariance(
            filename=filename,
            target_time=target_time,
            components=components
        )

    def getLocalPar(self, solverConfig, N, use_sigma=False):
        self._AdjustPathConfig = self.__adjust_config.PathOfFiles()
        self._AdjustPathConfig.__dict__.update(self.__adjust_config.PathOfFilesConfig.copy())

        '''get orbit par'''
        self.res_dir = solverConfig.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]

        self.b_rss = 0

        for i in tqdm(self.__arclist, desc='Solve local parameters: '):
            res_filename = self.res_dir + '/Orbit_' + str(i) + '_' + 'AB.hdf5'
            h5 = h5py.File(res_filename, 'r')
            self.__orbit_nominal = h5['r'][()]
            self.__t_nominal = h5['t'][()]
            self.__local_r = h5['local_r'][()]
            self.__global_r = h5['global_r'][()]
            h5.close()

            LocalA = self.getOrbitPar(
                sat=SatID.A, arcNo=i, use_sigma=use_sigma
            )
            LocalB = self.getOrbitPar(
                sat=SatID.B, arcNo=i, use_sigma=use_sigma
            )

            self.saveAccPar(LocalA, LocalB, arcNo=i)
            self.savePostKbrr(arcNo=i, use_sigma=use_sigma)

        if use_sigma:
            # The GLS normal matrix already contains the absolute covariance
            # scale through C^-1, so no additional OLS variance factor applies.
            cov = np.linalg.pinv(N)
        else:
            sigma2_hat = self.b_rss / (self.n - self.p)
            cov = sigma2_hat * np.linalg.inv(N)
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        CovName = solverConfig.CovarianceMatrix + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '.hdf5'
        if not os.path.exists(solverConfig.CovarianceMatrix):
            os.mkdir(solverConfig.CovarianceMatrix)
        h5 = h5py.File(CovName, 'w')
        h5.create_dataset('cov', data=cov)
        h5.close()

        return sigma

    def getOrbitPar(self, sat: SatID, arcNo: int, use_sigma=False):
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

        covariance = None
        if use_sigma:
            fit_epoch = np.rint(t_nominal).astype(np.int64) % 60 == 0
            epoch_index = np.flatnonzero(fit_epoch)
            fit_index = (
                3 * epoch_index[:, None] + np.arange(3)
            ).reshape(-1)
            covariance_name = orbit_covariance_filename(
                self.__solver_config.PostOrbit,
                self.__date_span,
                sat.name
            )
            covariance = self._load_arc_covariance(
                covariance_name, t_nominal[fit_epoch], components=3
            )
            Local, _ = self._solve_local_parameters(
                global_r[fit_index, :], local_r[fit_index, :],
                obs[fit_index, 0], self.cs, covariance
            )
            b_res = obs[:, 0] - global_r @ self.cs - local_r @ Local
        else:
            Local, b_res = self._solve_local_parameters(
                global_r, local_r, obs[:, 0], self.cs
            )

        self.b_rss += np.sum(b_res ** 2)

        res_path = self.__solver_config.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'
        if not os.path.exists(res_path):
            os.makedirs(res_path, exist_ok=True)
        res_name = res_path + 'Orbit_' + str(arcNo) + str(sat.name) + '.hdf5'
        h5 = h5py.File(res_name, 'w')
        h5.create_dataset('time', data=t_nominal)
        h5.create_dataset('post_Orbit', data=b_res)
        h5.close()
        return Local

    def getOrbitPar_V2(self, sat: SatID, arcNo: int):
        res_filename = self.res_dir + '/Orbit_' + str(arcNo) + '_' + 'AB.hdf5'
        designData = h5py.File(res_filename, 'r')

        new_index = designData['newindex'][()]
        t_nominal = designData['Orbit_t_%1s' % sat.name][()]
        obs = designData['Orbit_obs_%1s' % sat.name][()]
        local_r = designData['Orbit_local_%1s' % sat.name][()]
        global_r = designData['Orbit_global_%1s' % sat.name][()]
        designData.close()

        Local = np.linalg.lstsq(local_r, (obs[:, 0] - global_r @ self.cs), rcond=None)[0]

        b_res = obs[:, 0] - local_r @ Local - global_r @ self.cs

        self.b_rss += np.sum(b_res ** 2)

        res_path = self.__solver_config.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'
        if not os.path.exists(res_path):
            os.makedirs(res_path, exist_ok=True)
        res_name = res_path + 'Orbit_' + str(arcNo) + str(sat.name) + '.hdf5'
        h5 = h5py.File(res_name, 'w')
        h5.create_dataset('time', data=t_nominal)
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

    def savePostKbrr(self, arcNo: int, use_sigma=False):
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

        if obs.size == 0:
            print(
                'Arc {} has no KBRR observations after calibration; '
                'saving an empty posterior residual block'.format(arcNo)
            )

        sst_config = self.__neq_config.SST()
        sst_config.__dict__.update(self.__neq_config.SSTConfig.copy())
        if sst_config.CalibrateEmpiricalParameters:
            local_dm = np.concatenate(
                (local_dm_empirical, local_dm), axis=1
            )

        covariance = None
        if use_sigma:
            covariance_name = range_rate_covariance_filename(
                self.__solver_config.PostKBRR,
                self.__date_span
            )
            covariance = self._load_arc_covariance(
                covariance_name, t_dm, components=1
            )

        Local_par, kbrr_res = self._solve_local_parameters(
            global_dm, local_dm, obs, self.cs, covariance
        )
        scale_parameter_count = 18 if self.AccelerometerConfig.isScale else 0
        geofit_kbrr = self._gravity_field_fit(
            global_dm, self.cs, scale_parameter_count
        )

        res_path = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'
        if not os.path.exists(res_path):
            os.makedirs(res_path, exist_ok=True)

        res_name = res_path + 'RangeRate_' + str(arcNo) + '.hdf5'
        h5 = h5py.File(res_name, 'w')
        h5.create_dataset('kbrr_par', data=Local_par)
        h5.create_dataset('time', data=t_dm)
        h5.create_dataset('post_kbrr', data=kbrr_res)
        h5.create_dataset('geofit_kbrr', data=geofit_kbrr)
        h5.close()

        standard_kbrr_directory = getattr(
            self._AdjustPathConfig,
            'StandardKBRRTemp',
            StandardKBRR.output_directory(self._AdjustPathConfig.RangeRateTemp)
        )
        orbit_a = self._rd.getData(
            arc=arcNo,
            kind=Payload.GNV,
            sat=SatID.A
        )[:, 0:4]
        orbit_b = self._rd.getData(
            arc=arcNo,
            kind=Payload.GNV,
            sat=SatID.B
        )[:, 0:4]
        eop = EOP().configure(frameConfig=self.__frame_config).load()
        frame = Frame(eop).configure(frameConfig=self.__frame_config)
        longitude, latitude, altitude = StandardKBRR.midpoint_coordinates(
            gps_time=t_dm,
            orbit_a=orbit_a,
            orbit_b=orbit_b,
            frame=frame
        )
        for stage, range_rate in (
                ('Pre', obs), ('Post', kbrr_res),
                ('Geofit', geofit_kbrr)):
            StandardKBRR().configure(
                filedir=standard_kbrr_directory,
                date_span=self.__date_span,
                stage=stage,
                mission=self.__interface_config.mission
            ).save_arc(
                gps_time=t_dm,
                longitude=longitude,
                latitude=latitude,
                altitude=altitude,
                range_rate=range_rate,
                arc_no=arcNo
            )
        pass

    @staticmethod
    def _gravity_field_fit(global_dm, global_parameters,
                           scale_parameter_count=0):
        """Return the KBRR contribution of spherical-harmonic parameters."""
        design = np.asarray(global_dm, dtype=np.float64)
        parameters = np.asarray(global_parameters, dtype=np.float64).reshape(-1)
        scale_parameter_count = int(scale_parameter_count)
        if design.ndim != 2 or design.shape[1] != parameters.size:
            raise ValueError(
                'Geofit dimensions are inconsistent: design={}, parameters={}'
                .format(design.shape, parameters.shape)
            )
        if not 0 <= scale_parameter_count < parameters.size:
            raise ValueError(
                'Invalid scale parameter count for geofit: %d'
                % scale_parameter_count
            )
        return (
            design[:, scale_parameter_count:]
            @ parameters[scale_parameter_count:]
        )

    def savePostKbrr_V2(self, arcNo: int):
        '''get kbrr par'''
        dm_dir = self.__solver_config.DesignMatrixTemp + '/' + self.__date_span[0] + '_' + self.__date_span[1]
        res_filename = dm_dir + '/' + SSTObserve.RangeRate.name + '_' + str(arcNo) + '.hdf5'
        designData = h5py.File(res_filename, 'r')
        t_dm = designData['%s_t' % SSTObserve.RangeRate.name][()]
        global_dm = designData['%s_global' % SSTObserve.RangeRate.name][()]
        local_dm = designData['%s_local' % SSTObserve.RangeRate.name][()]
        obs = designData['%s_obs' % SSTObserve.RangeRate.name][()]
        designData.close()

        Local_par = np.linalg.lstsq(local_dm, (obs[:, 0] - global_dm @ self.cs))[0]

        kbrr_res = obs[:, 0] - local_dm @ Local_par - global_dm @ self.cs

        res_path = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1] + '/'
        if not os.path.exists(res_path):
            os.makedirs(res_path, exist_ok=True)

        res_name = res_path + 'RangeRate_' + str(arcNo) + '.hdf5'
        h5 = h5py.File(res_name, 'w')
        h5.create_dataset('kbrr_par', data=Local_par)
        h5.create_dataset('time', data=t_dm)
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

    def GLS(self, orbit_fixed_diagonal=0.004):
        '''get orbit obs'''
        orbit_path = self.__solver_config.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1]
        kbrr_path = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1]
        self.lagNum = int(self.__interface_config.arc_length * 3600 / self.__interface_config.sr_target)

        kbrr = []
        orbit_a = []
        orbit_b = []
        time_a = []
        time_b = []
        time_kbrr = []
        for i in tqdm(self.__arclist, desc='Solve Orbit and KBRR Cov: '):
            postOrbitname_a = orbit_path + '/Orbit_' + str(i) + 'A.hdf5'
            postOrbitname_b = orbit_path + '/Orbit_' + str(i) + 'B.hdf5'
            kbrrname = kbrr_path + '/' + SSTObserve.RangeRate.name + '_' + str(i) + '.hdf5'

            with h5py.File(postOrbitname_a, "r") as postOrbith5_a, \
                    h5py.File(postOrbitname_b, "r") as postOrbith5_b, \
                    h5py.File(kbrrname, "r") as postKBRRh5:
                time_orbit_a = postOrbith5_a['time'][:]
                time_orbit_b = postOrbith5_b['time'][:]
                time_kbr = postKBRRh5['time'][:]

                res_orbit_a = postOrbith5_a['post_Orbit'][:]
                res_orbit_b = postOrbith5_b['post_Orbit'][:]
                res_kbrr = postKBRRh5['post_kbrr'][:]

            time_a.extend(time_orbit_a)
            time_b.extend(time_orbit_b)
            time_kbrr.extend(time_kbr)

            orbit_a.extend(res_orbit_a)
            orbit_b.extend(res_orbit_b)
            kbrr.extend(res_kbrr)

        kbrr_acov, kbrr_dt = self.getKBRRSigma(
            time=time_kbrr, res=kbrr, return_autocovariance=True
        )
        save_stationary_covariance(
            range_rate_covariance_filename(
                self.__solver_config.PostKBRR, self.__date_span
            ),
            autocovariance=kbrr_acov,
            sample_interval=kbrr_dt,
            time_origin=float(np.min(time_kbrr))
        )
        save_stationary_covariance(
            orbit_covariance_filename(
                self.__solver_config.PostOrbit, self.__date_span, 'A'
            ),
            diagonal_value=orbit_fixed_diagonal
        )
        save_stationary_covariance(
            orbit_covariance_filename(
                self.__solver_config.PostOrbit, self.__date_span, 'B'
            ),
            diagonal_value=orbit_fixed_diagonal
        )
        pass

    def GLS_full(self):
        '''get orbit obs'''
        orbit_path = self.__solver_config.PostOrbit + '/' + self.__date_span[0] + '-' + self.__date_span[1]
        kbrr_path = self.__solver_config.PostKBRR + '/' + self.__date_span[0] + '-' + self.__date_span[1]
        self.lagNum = int(self.__interface_config.arc_length * 3600 / self.__interface_config.sr_target)

        kbrr = []
        orbit_a = []
        orbit_b = []
        time_a = []
        time_b = []
        time_kbrr = []
        for i in tqdm(self.__arclist, desc='Solve Orbit and KBRR Cov: '):
            postOrbitname_a = orbit_path + '/Orbit_' + str(i) + 'A.hdf5'
            postOrbitname_b = orbit_path + '/Orbit_' + str(i) + 'B.hdf5'
            kbrrname = kbrr_path + '/' + SSTObserve.RangeRate.name + '_' + str(i) + '.hdf5'

            with h5py.File(postOrbitname_a, "r") as postOrbith5_a, \
                    h5py.File(postOrbitname_b, "r") as postOrbith5_b, \
                    h5py.File(kbrrname, "r") as postKBRRh5:
                time_orbit_a = postOrbith5_a['time'][:]
                time_orbit_b = postOrbith5_b['time'][:]
                time_kbr = postKBRRh5['time'][:]

                res_orbit_a = postOrbith5_a['post_Orbit'][:]
                res_orbit_b = postOrbith5_b['post_Orbit'][:]
                res_kbrr = postKBRRh5['post_kbrr'][:]

            time_a.extend(time_orbit_a)
            time_b.extend(time_orbit_b)
            time_kbrr.extend(time_kbr)

            orbit_a.extend(res_orbit_a)
            orbit_b.extend(res_orbit_b)
            kbrr.extend(res_kbrr)

        kbrr_acov, kbrr_dt = self.getKBRRSigma(
            time=time_kbrr, res=kbrr, return_autocovariance=True
        )
        orbit_acov_a, orbit_dt_a = self.getOrbitSigma(
            time=time_a, res=orbit_a, return_autocovariance=True
        )
        orbit_acov_b, orbit_dt_b = self.getOrbitSigma(
            time=time_b, res=orbit_b, return_autocovariance=True
        )

        save_stationary_covariance(
            range_rate_covariance_filename(
                self.__solver_config.PostKBRR, self.__date_span
            ),
            autocovariance=kbrr_acov,
            sample_interval=kbrr_dt,
            time_origin=float(np.min(time_kbrr))
        )
        save_stationary_covariance(
            orbit_covariance_filename(
                self.__solver_config.PostOrbit, self.__date_span, 'A'
            ),
            autocovariance=orbit_acov_a,
            sample_interval=orbit_dt_a,
            time_origin=float(np.min(time_a)),
            allow_nearest_epoch=True
        )
        save_stationary_covariance(
            orbit_covariance_filename(
                self.__solver_config.PostOrbit, self.__date_span, 'B'
            ),
            autocovariance=orbit_acov_b,
            sample_interval=orbit_dt_b,
            time_origin=float(np.min(time_b)),
            allow_nearest_epoch=True
        )
        pass

    def getSigma(self, time, res):
        n = len(res)
        res = np.array(res)
        res = detrend(res, type="linear")
        res -= res.mean()
        f, P_res = welch(res, fs=0.2, nperseg=len(res)//8)
        plt.loglog(f, P_res, label='PSD from residual-kbrr')
        plt.legend()
        plt.grid()
        plt.show()
        # res = detrend(res)
        # res -= np.mean(res)

        # p_max = min(50, n // 10)
        # p = self.pick_AR_order(res, p_max=p_max, ic='bic')
        #
        # phi, sigma2 = self.estimate_ARp(res, p)
        # cov = self.AR_toeplitz_cov(phi, sigma2, n)

        # res = detrend(res, type='linear')
        # acov = self.estimate_acov(res, max_lag=self.lagNum - 1)

        time = np.asarray(time)
        dt = 5
        t0 = time[0]

        # 映射为整数采样点
        idx = np.round((time - t0) / dt).astype(np.int64)

        # 构建稀疏等间隔序列
        L = idx.max() + 1
        x = np.zeros(L)
        mask = np.zeros(L)
        x[idx] = res
        mask[idx] = 1.0

        acov, _ = self.estimate_acov_by_lag(x=x, mask=mask, max_lag=self.lagNum - 1)
        plt.plot(acov)
        plt.title("Autocovariance")
        plt.xlabel("Lag")
        plt.grid()
        plt.show()
        # FFT
        eps = max(acov[0] * 1e-10, 1e-18)
        # eps = max(acov[0] * 1e-8, 1e-20)
        # Step 1: DCT-I
        S = dct(acov, type=1, norm=None)
        # dm = self.dct_matrix_matlab_style(len(acov))
        # S = dm @ acov
        #
        k = np.arange(len(acov))
        f = k / (2 * (len(acov) - 1) * 5)
        plt.loglog(f, S, label='PSD from acov (DCT)')
        plt.axhline(eps, color='r', ls='--', label='eps')
        plt.legend()
        plt.grid()
        plt.show()
        #
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
        acov_corr = idct(S_clipped, type=1, norm=None)
        # acov_corr /= (2 * (len(acov) - 1))
        Sigma = toeplitz(acov_corr)
        return Sigma

    def getSigma2(self, time, res):
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

        # f, P_res = welch(res_x, fs=0.2, nperseg=len(res_x)//8)
        # plt.loglog(f, P_res, label='PSD from residual')
        # plt.legend()
        # plt.grid()
        # plt.show()
        # acov_x = self.estimate_acov(res_x, max_lag=self.lagNum - 1)
        res_x = np.asarray(res_x)
        time = np.asarray(time)
        dt = 5
        t0 = time[0]

        # 映射为整数采样点
        idx = np.round((time - t0) / dt).astype(np.int64)

        # 构建稀疏等间隔序列
        L = idx.max() + 1

        x = np.zeros(L)
        mask = np.zeros(L)
        x[idx] = res_x
        mask[idx] = 1.0

        y = np.zeros(L)
        y[idx] = res_y

        z = np.zeros(L)
        z[idx] = res_z

        acov_x, _ = self.estimate_acov_by_lag(x=x, mask=mask, max_lag=self.lagNum - 1)
        acov_y, _ = self.estimate_acov_by_lag(x=y, mask=mask, max_lag=self.lagNum - 1)
        acov_z, _ = self.estimate_acov_by_lag(x=z, mask=mask, max_lag=self.lagNum - 1)
        # acov_y = self.estimate_acov(res_y, max_lag=self.lagNum - 1)
        # acov_z = self.estimate_acov(res_z, max_lag=self.lagNum - 1)

        # plt.plot(acov_x)
        # plt.title("Autocovariance")
        # plt.xlabel("Lag")
        # plt.grid()
        # plt.show()
        #
        # plt.plot(acov_z)
        # plt.title("Autocovariance")
        # plt.xlabel("Lag")
        # plt.grid()
        # plt.show()
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
        eps_x = max(acov_x[0] * 1e-10, 1e-12)
        #
        # # acov_full = self.psd_clip_autocov(acov, n, eps=eps)
        #
        # # Sigma = toeplitz(acov_full)
        # '''正定�?''
        # # Sigma += np.eye(n) * 1e-2
        # # Sigma = self.nearest_pos_def(Sigma, eps_rel=1e-10)
        #
        # Step 1: DCT-I
        S_x = dct(acov_x, type=1, norm=None)

        k = np.arange(len(acov_x))
        f = k / (2 * (len(acov_x) - 1) * 5)
        plt.loglog(f, S_x, label='PSD from acov (DCT)')
        plt.axhline(eps_x, color='r', ls='--', label='eps')
        plt.legend()
        plt.grid()
        plt.show()
        # f = np.linspace(0, 1/(2 * 5), len(S_x))
        # plt.figure(figsize=(7, 4))
        # plt.loglog(f[1:4320], S_x[1:], lw=1.5)
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("PSD (variance)")
        # plt.title("Noise PSD estimated via DCT-I")
        # plt.grid(True)

        # plt.tight_layout()
        # plt.show()
        # Step 2: clipping
        S_clipped_x = np.maximum(S_x, eps_x)

        # Step 3: inverse DCT-I
        acov_corr_x = idct(S_clipped_x, type=1, norm=None)
        # acov_corr_x /= (2 * (len(acov_x) - 1))
        # plt.plot(acov_corr_x)
        # plt.title("Autocovariance-idct")
        # plt.xlabel("Lag")
        # plt.grid()
        # plt.show()
        # plt.plot(np.arange(len(acov_corr)), acov_corr)
        # plt.show()
        Sigma_x = toeplitz(acov_corr_x)

        # '''FFT'''
        eps_y = max(acov_y[0] * 1e-10, 1e-12)
        S_y= dct(acov_y, type=1, norm=None)
        S_clipped_y = np.maximum(S_y, eps_y)
        acov_corr_y = idct(S_clipped_y, type=1, norm=None)
        # acov_corr_y /= (2 * (len(acov_y) - 1))
        Sigma_y = toeplitz(acov_corr_y)

        # '''FFT'''
        eps_z = max(acov_z[0] * 1e-10, 1e-12)
        S_z = dct(acov_z, type=1, norm=None)
        S_clipped_z = np.maximum(S_z, eps_z)
        acov_corr_z = idct(S_clipped_z, type=1, norm=None)
        # acov_corr_z /= (2 * (len(acov_z) - 1))
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
        S = dct(acov, type=1, norm=None)

        k = np.arange(len(acov))
        f = k / (2 * (len(acov) - 1) * 5)
        P_acov = S.real
        P_acov /= P_acov.max()  # 只比较形�?
        plt.loglog(f, P_acov, label='PSD from acov (DCT)')
        plt.legend()
        plt.grid()
        plt.show()

        # Step 2: clipping
        S_clipped = np.maximum(S, eps)
        # Step 3: inverse DCT-I
        acov_corr = idct(S_clipped, type=1, norm=None)
        Sigma = toeplitz(acov_corr)
        return Sigma

    def getSigma6(self, acov_x, acov_y, acov_z):

        eps_x = max(acov_x[0] * 1e-5, 1e-11)
        #
        # # acov_full = self.psd_clip_autocov(acov, n, eps=eps)
        #
        # # Sigma = toeplitz(acov_full)
        # '''正定�?''
        # # Sigma += np.eye(n) * 1e-2
        # # Sigma = self.nearest_pos_def(Sigma, eps_rel=1e-10)
        #
        # Step 1: DCT-I
        S_x = dct(acov_x, type=1, norm=None)
        k = np.arange(len(acov_x))
        f = k / (2 * (len(acov_x) - 1) * 5)
        # P_acov = S_x.real
        # P_acov /= P_acov.max()  # 只比较形�?
        # plt.loglog(f, P_acov, label='PSD from acov (DCT)')
        # plt.legend()
        # plt.grid()
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

    def getKBRRSigma(self, time, res, return_autocovariance=False):
        n = len(res)
        res = np.array(res)
        res = detrend(res, type="linear")
        res -= res.mean()

        # f, P_res = welch(res, fs=0.2, nperseg=len(res) // 8)
        # plt.loglog(f, P_res, label='PSD from residual-kbrr')
        # plt.legend()
        # plt.grid()
        # plt.show()

        sample_interval = float(self.__interface_config.sr_target)
        t_uniform, res_filled = self.fill_gaps_with_interp_and_psd(
            np.array(time), res, sample_interval)
        acov = self.estimate_acov_stable(res_filled, max_lag=len(res_filled) - 1)

        # tau, acov, Npair = self.autocorr_with_gaps_fft(time, res, 5, max_lag=4320)

        # _, fill_data = self.fill_all_gaps(time=np.array(time), residuals=res, sr=5, max_lag=1000)
        # acov = self.estimate_acov(fill_data, max_lag=self.lagNum - 1)
        # tau, acov = self.autocorr_with_gaps(time, res, 5, 21600)
        # plt.plot(acov)
        # plt.title("Autocovariance")
        # plt.xlabel("Lag")
        # plt.grid()
        # plt.show()
        # FFT
        eps = max(acov[0] * 1e-4, 1e-18)
        # eps = max(acov[0] * 1e-8, 1e-20)
        # Step 1: DCT-I
        S = dct(acov, type=1, norm=None)
        #
        # k = np.arange(len(acov))
        # f = k / (2 * (len(acov) - 1) * 5)
        # plt.loglog(f, S, label='PSD from acov (DCT)')
        # plt.axhline(eps, color='r', ls='--', label='eps')
        # plt.legend()
        # plt.grid()
        # plt.show()
        #
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
        acov_corr = idct(S_clipped, type=1, norm=None)
        # acov_corr /= (2 * (len(acov) - 1))
        if return_autocovariance:
            lag_count = min(self.lagNum, len(acov_corr))
            return acov_corr[:lag_count], sample_interval
        # Sigma = toeplitz(acov_corr)
        Sigma = self.toeplitz_cov_from_acov(acov_corr, self.lagNum)
        return Sigma

    def getOrbitSigma(self, time, res, return_autocovariance=False):
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

        # f1, P_res1 =  self.psd_no_window(residual=res_x, fs=0.2)
        # plt.figure(figsize=(7, 5))
        # plt.loglog(f1, P_res1, label='Residual FFT PSD')
        # # plt.loglog(f2, P_res2, label='whlch PSD')
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel('PSD')
        # plt.ylim(1e-13, 2)
        # plt.legend()
        # plt.grid(True, which='both')
        # plt.tight_layout()
        # plt.show()
        # acov_x = self.estimate_acov(res_x, max_lag=self.lagNum - 1)
        # time = np.asarray(time)

        sample_interval = float(self.__solver_config.OrbitFre)
        covariance_result = self.cal_toeplitz_cov(
            time, res_x, res_y, res_z,
            dt=sample_interval,
            return_autocovariance=return_autocovariance
        )
        if return_autocovariance:
            return covariance_result
        Sigma_x, Sigma_y, Sigma_z = covariance_result
        # Sigma_x2, Sigma_y2, Sigma_z2 = self.cal_toeplitz_cov(time, res_x, res_y, res_z, dt=5)

        # Sigma_x = (Sigma_x1 + Sigma_x2) / 2
        # Sigma_y = (Sigma_y1 + Sigma_y2) / 2
        # Sigma_z = (Sigma_z1 + Sigma_z2) / 2

        N = len(Sigma_x)
        Sigma = np.zeros((3 * N, 3 * N))

        for i in range(N):
            for j in range(N):
                Sigma[3 * i, 3 * j] = Sigma_x[i, j]
                Sigma[3 * i + 1, 3 * j + 1] = Sigma_y[i, j]
                Sigma[3 * i + 2, 3 * j + 2] = Sigma_z[i, j]

        return Sigma
        # return Sigma_x, Sigma_y, Sigma_z

    def getOrbitSigma2(self, time, res, diagonal_value=0.004):
        """Return the configured fixed diagonal orbit covariance."""
        diagonal_value = float(diagonal_value)
        if not np.isfinite(diagonal_value) or diagonal_value <= 0.0:
            raise ValueError('orbit covariance diagonal must be positive')
        return diagonal_value * np.eye(3 * self.lagNum)

    def estimate_acov(self, residuals, max_lag=50):
        """
        用残差估计自协函数（ACF），无偏估计版本�?
        """
        residuals = np.asarray(residuals)
        residuals = residuals - residuals.mean()
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

        # 只提取最小的 k 个特征�?
        vals, vecs = eigsh(mat, k=20, which='SA')

        min_val = np.min(vals)
        if min_val > eps_rel:
            return mat  # 已正�?

        # 修正负特征�?
        vals_clipped = np.maximum(vals, eps_rel)

        # 只更新部分特征值贡献（低秩修补�?
        Sigma_fix = mat + vecs @ np.diag(vals_clipped - vals) @ vecs.T

        return Sigma_fix

    def psd_clip_autocov(self, acov, n_out=None, eps=1e-20):
        """
            Method A: FFT �?PSD clipping �?IFFT.
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
    # def estimate_ARp(self, res, p):
    #     # 去均�?
    #     r = res - np.mean(res)
    #     # Yule-Walker 估计 AR(p)
    #     rho, sigma = yule_walker(r, order=p, method='mle')
    #     return rho, sigma

    def build_banded_toeplitz(self, acov, N, K, eps_factor=1e-8):
        """
        acov       : 自协函数 γ(0..)
        N          : 弧段长度（如 4320�?
        K          : 最大滞后（带宽�?
        eps_factor : nugget 强度
        """
        assert K < N
        assert len(acov) >= K + 1

        Sigma = np.zeros((N, N))

        # 主对�?+ K 条副对角
        for k in range(K + 1):
            val = acov[k]
            if k == 0:
                Sigma += np.diag(val * np.ones(N))
            else:
                Sigma += np.diag(val * np.ones(N - k), k)
                Sigma += np.diag(val * np.ones(N - k), -k)

        # 强制对称（数值安全）
        Sigma = 0.5 * (Sigma + Sigma.T)

        # 对角加载（nugget�?
        eps = eps_factor * acov[0]
        Sigma += eps * np.eye(N)

        return Sigma

    def saveSigma1(self, Sigma, time, h5path):

        if "Sigma" in h5path:
            del h5path["Sigma"]
        if "time" in h5path:
            del h5path["time"]
        #
        # if "Sigma_y" in h5path:
        #     del h5path["Sigma_y"]
        # #
        # if "Sigma_z" in h5path:
        #     del h5path["Sigma_z"]
        #
        # h5path.create_dataset("Sigma_x", data=Sigma_x)
        # h5path.create_dataset("Sigma_y", data=Sigma_y)
        h5path.create_dataset("time", data=time)
        h5path.create_dataset("Sigma", data=Sigma)
        return self

    def saveSigma2(self, Sigma, h5path):

        if "Sigma" in h5path:
            del h5path["Sigma"]

        h5path.create_dataset("Sigma", data=Sigma)
        h5path.close()
        return self

    def dct_matrix_matlab_style(self, n):
        """
        严格等价于你给出�?MATLAB / Fortran DCT-I 实现
        """

        # 定义 pi
        pi = np.pi

        # 分配矩阵
        dctmat = np.zeros((n, n), dtype=np.float64)

        # ===== Step 1: 构�?2*cos(pi*(i-1)*(j-1)/(n-1)) =====
        for i in range(n):          # i = 0 ... n-1
            for j in range(n):      # j = 0 ... n-1
                dctmat[i, j] = 2.0 * np.cos(
                    pi * i * j / (n - 1)
                )

        # ===== Step 2: 第一列和最后一列乘 0.5 =====
        dctmat[:, 0]  *= 0.5
        dctmat[:, -1] *= 0.5

        # ===== Step 3: 整体归一�?=====
        scale_factor = 1.0 / np.sqrt(2.0 * (n - 1))
        dctmat *= scale_factor

        return dctmat

    def estimate_acov_with_gaps(self, t, r, dt, max_lag):
        """
        向量�?+ 支持 gap 的自协函数估�?

        Parameters
        ----------
        t : (N,) ndarray
            时间戳（秒）
        r : (N,) ndarray
            残差
        dt : float
            采样间隔
        max_lag : int
            最�?lag（点数）

        Returns
        -------
        acov : (max_lag+1,) ndarray
        counts : (max_lag+1,) ndarray
        """

        t = np.asarray(t)
        r = np.asarray(r)

        # 1️⃣ 映射到理想采样索�?
        n = np.round((t - t[0]) / dt).astype(int)

        # 2️⃣ 初始�?
        acov = np.zeros(max_lag + 1)
        counts = np.zeros(max_lag + 1, dtype=int)

        # 3️⃣ 对每�?lag 向量化计�?
        for k in range(max_lag + 1):
            idx_i = np.where(n + k <= n[-1])[0]
            idx_j = idx_i + np.searchsorted(n, n[idx_i] + k)

            # 保证索引合法
            valid = idx_j < len(n)
            idx_i = idx_i[valid]
            idx_j = idx_j[valid]

            # 检查是否真的差 k（排�?gap�?
            good = (n[idx_j] - n[idx_i]) == k
            idx_i = idx_i[good]
            idx_j = idx_j[good]

            if len(idx_i) == 0:
                continue

            acov[k] = np.dot(r[idx_i], r[idx_j])
            counts[k] = len(idx_i)

        acov[counts > 0] /= counts[counts > 0]
        acov[counts == 0] = 0.0

        return acov, counts

    def estimate_acov_by_lag(self, x, mask, max_lag=None):
        n = len(x)
        nfft = 1 << (2 * n - 1).bit_length()

        X = np.fft.fft(x, nfft)
        M = np.fft.fft(mask, nfft)

        # 分子：�?r_i r_{i+k}
        num = np.fft.ifft(X * np.conj(X)).real

        # 分母：�?1（有效配对数�?
        den = np.fft.ifft(M * np.conj(M)).real

        if max_lag is None:
            max_lag = n - 1

        acov = num[:max_lag + 1] / den[:max_lag + 1]
        acov[den[:max_lag + 1] == 0] = np.nan

        return acov, den[:max_lag + 1]

    def time_to_index(self, time, sr):
        """
        time : array_like, �?
        sr   : 采样间隔（秒�?
        """
        t0 = time[0]
        index = np.round((time - t0) / sr).astype(np.int64)
        return index

    def find_gaps(self, index):
        """
        index : 连续时间轴上的整�?index
        返回�?
            gaps = [(gap_start_idx, gap_end_idx, gap_length), ...]
        """
        gaps = []
        diff = np.diff(index)

        # gap 出现�?diff > 1
        gap_pos = np.where(diff > 1)[0]

        for p in gap_pos:
            gap_start = index[p] + 1
            gap_end = index[p + 1] - 1
            gap_len = gap_end - gap_start + 1
            gaps.append((gap_start, gap_end, gap_len))

        return gaps

    def build_indexed_series(self, index, residuals):
        """
        构建 dict: index -> residual
        """
        return dict(zip(index, residuals), type=1, norm=None)

    def fill_gap_colored_noise(self, acov, gap_len, eps=1e-12):
        """
        生成长度�?gap_len �?colored noise
        """
        C = self.toeplitz_cov_from_acov(acov, gap_len)

        # 数值稳�?
        C += eps * np.eye(gap_len)

        L = cholesky(C, lower=True)
        z = self._gls_rng.standard_normal(gap_len)

        return L @ z

    def fill_all_gaps(self, time, residuals, sr, max_lag):
        """
        返回�?
            full_index
            full_residuals（等间隔、连续）
        """
        # 1. index
        index = self.time_to_index(time, sr)

        # 2. gap detection
        gaps = self.find_gaps(index)

        # 3. residual dict
        res_dict = self.build_indexed_series(index, residuals)

        # 4. estimate acov from all residuals
        acov = self.estimate_acov(residuals, max_lag)

        # 5. full index range
        full_index = np.arange(index.min(), index.max() + 1)
        full_residuals = np.empty(len(full_index))

        ptr = 0
        for i, idx in enumerate(full_index):
            if idx in res_dict:
                full_residuals[i] = res_dict[idx]
            else:
                # 属于某个 gap �?延后统一�?
                full_residuals[i] = np.nan

        # 6. 填充每个 gap
        for gap_start, gap_end, gap_len in gaps:
            if gap_len > max_lag:
                fill = self._gls_rng.normal(
                    scale=np.sqrt(acov[0]),
                    size=gap_len
                )
            else:
                fill = self.fill_gap_colored_noise(acov, gap_len)

            mask = (full_index >= gap_start) & (full_index <= gap_end)
            full_residuals[mask] = fill

        return full_index, full_residuals

    def build_uniform_time_and_mask(self, t, r, dt):
        """
        t : 原始时间（不连续�?
        r : 残差
        dt: 采样间隔
        """
        t = np.asarray(t)
        r = np.asarray(r)

        t0 = t.min()
        idx = np.round((t - t0) / dt).astype(int)

        N = idx.max() + 1

        r_full = np.full(N, np.nan)
        r_full[idx] = r

        t_full = t0 + np.arange(N) * dt

        mask_valid = ~np.isnan(r_full)

        return t_full, r_full, mask_valid

    def low_freq_interpolate(self, t_full, r_full, mask_valid):
        t_valid = t_full[mask_valid]
        r_valid = r_full[mask_valid]

        interp_func = interp1d(
            t_valid,
            r_valid,
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False
        )

        r_low = interp_func(t_full)
        return r_low

    def estimate_psd_welch(self, r, dt, nperseg=1024):
        fs = 1.0 / dt
        f, Pxx = welch(
            r,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend='constant',
            scaling='density'
        )
        return f, Pxx

    def generate_colored_noise_from_psd(self, f, Pxx, N, dt):
        """
        �?PSD 生成长度�?N 的随机噪�?
        """
        df = f[1] - f[0]

        amp = np.sqrt(Pxx * df)
        phase = self._gls_rng.uniform(
            0, 2 * np.pi, size=len(amp)
        )

        spectrum = amp * np.exp(1j * phase)

        # 构造共轭对称谱
        full_spectrum = np.zeros(N, dtype=complex)
        n_half = len(spectrum)

        full_spectrum[:n_half] = spectrum
        full_spectrum[-n_half:] = np.conj(spectrum[::-1])

        noise = np.fft.ifft(full_spectrum).real
        return noise

    def fill_gaps_with_interp_and_psd(self, t, r, dt):
        t_full, r_full, mask_valid = self.build_uniform_time_and_mask(t, r, dt)

        # 低频部分
        r_low = self.low_freq_interpolate(t_full, r_full, mask_valid)

        # 高频 PSD 用“有效残�?- 低频�?
        hf_res = r_full[mask_valid] - r_low[mask_valid]

        f, Pxx = self.estimate_psd_welch(hf_res, dt)

        # 生成全序列高频噪�?
        noise = self.generate_colored_noise_from_psd(f, Pxx, len(r_full), dt)

        # 仅在 gap 处加噪声
        r_filled = r_low.copy()
        r_filled[~mask_valid] += noise[~mask_valid]

        # 有效点保持原始数�?
        r_filled[mask_valid] = r_full[mask_valid]

        return t_full, r_filled

    def estimate_acov_stable(self, res, max_lag):
        res = np.asarray(res)
        res = res - np.mean(res)

        n = len(res)
        # max_lag = min(max_lag, n // 5)

        acov = np.zeros(max_lag + 1)
        for k in range(max_lag + 1):
            acov[k] = np.dot(res[:n - k], res[k:]) / (n - k)

        # taper（非常关键）
        window = np.hanning(2 * max_lag + 1)[max_lag:]
        acov *= window

        return acov

    def toeplitz_cov_from_acov(self, acov, N):
        if len(acov) < N:
            acov = np.pad(acov, (0, N - len(acov)), mode='constant')
        return toeplitz(acov[:N])

    def smooth_psd_logpoly(self, f, S, deg=2, quantile=0.2):
        """
        �?log-log 多项式拟�?PSD，自动忽略极小�?
        """
        f = np.asarray(f)
        S = np.asarray(S)

        mask = (f > 0) & (S > 0)
        f = f[mask]
        S = S[mask]

        logf = np.log(f)
        logS = np.log(S)

        # 剔除极小值（分位数，而不�?eps�?
        thresh = np.quantile(logS, quantile)
        fit_mask = logS > thresh

        coeff = np.polyfit(logf[fit_mask], logS[fit_mask], deg)
        logS_fit = np.polyval(coeff, logf)

        S_smooth = np.exp(logS_fit)
        return S_smooth

    def smooth_log_psd(self, S, sigma=3):
        logS = np.log(np.maximum(S, 1e-300))
        logS_smooth = gaussian_filter1d(logS, sigma=sigma)
        return np.exp(logS_smooth)

    def autocorr_time_domain(self,
            t,
            r,
            dt,
            max_lag,
            normalize=True
    ):
        """
        Time-domain autocorrelation for irregular time series.

        Parameters
        ----------
        t : (N,) array
            Time stamps (seconds), must be sorted
        r : (N,) array
            Residuals
        dt : float
            Nominal sampling interval (e.g. 1.0)
        max_lag : float
            Maximum lag time (seconds)
        normalize : bool
            If True, normalize by R(0)

        Returns
        -------
        tau : (K+1,) array
            Lag times
        R : (K+1,) array
            Autocorrelation values
        Npair : (K+1,) array
            Number of contributing pairs
        """

        t = np.asarray(t, dtype=float)
        r = np.asarray(r, dtype=float)

        # sort safeguard
        idx = np.argsort(t)
        t = t[idx]
        r = r[idx]

        # remove mean
        r = r - np.mean(r)

        K = int(np.floor(max_lag / dt))
        tau = np.arange(K + 1) * dt

        sum_corr = np.zeros(K + 1)
        count = np.zeros(K + 1, dtype=int)

        N = len(t)

        for i in range(N):
            ti = t[i]
            ri = r[i]

            j = i
            while j < N:
                dtij = t[j] - ti
                if dtij > max_lag + dt / 2:
                    break

                k = int(np.round(dtij / dt))
                if 0 <= k <= K and abs(dtij - k * dt) <= dt / 2:
                    sum_corr[k] += ri * r[j]
                    count[k] += 1

                j += 1

        R = np.zeros_like(sum_corr)
        valid = count > 0
        R[valid] = sum_corr[valid] / count[valid]

        if normalize and R[0] != 0:
            R /= R[0]

        return tau, R, count

    def autocorr_with_gaps_fft(self, t, r, dt, max_lag, normalize=True):
        """
        Fast autocorrelation with gaps using FFT and mask correction.

        Parameters
        ----------
        t : (N,) array
            Time stamps (seconds, integer or near-integer)
        r : (N,) array
            Residuals
        dt : float
            Sampling interval (seconds)
        max_lag : int
            Maximum lag in seconds
        normalize : bool

        Returns
        -------
        tau : array
            Lag times
        R : array
            Autocorrelation
        Npair : array
            Effective number of sample pairs
        """

        t = np.asarray(t)
        r = np.asarray(r)

        # build uniform grid
        t0 = int(np.round(t.min()))
        t1 = int(np.round(t.max()))
        N = int((t1 - t0) / dt) + 1

        # index mapping
        idx = ((t - t0) / dt).round().astype(int)

        x = np.zeros(N)
        m = np.zeros(N)

        r = r - np.mean(r)

        x[idx] = r
        m[idx] = 1.0

        # zero padding for linear convolution
        nfft = 1 << (2 * N - 1).bit_length()

        Fx = rfft(x, nfft)
        Fm = rfft(m, nfft)

        corr_x = irfft(Fx * np.conj(Fx), nfft)
        corr_m = irfft(Fm * np.conj(Fm), nfft)

        corr_x = corr_x[:max_lag + 1]
        corr_m = corr_m[:max_lag + 1]

        R = np.zeros_like(corr_x)
        valid = corr_m > 0
        R[valid] = corr_x[valid] / corr_m[valid]

        if normalize and R[0] != 0:
            R /= R[0]

        tau = np.arange(len(R)) * dt

        return tau, R, corr_m

    import numpy as np

    def psd_no_window(self, residual, fs):
        """
        严谨的单�?PSD（不加窗，不平滑�?
        """
        r = np.asarray(residual, dtype=float)
        r = r - np.mean(r)  # 必须去均值（否则 DC 污染低频�?
        N = len(r)

        # FFT
        R = np.fft.fft(r)
        freqs = np.fft.fftfreq(N, d=1 / fs)

        # PSD 定义（Parseval 一致）
        psd = (np.abs(R) ** 2) / (fs * N)

        # 单边�?
        idx = freqs >= 0
        freqs = freqs[idx]
        psd = psd[idx]

        # �?DC / Nyquist 频率能量翻�?
        psd[1:-1] *= 2

        return freqs, psd

    def autocorr_with_gaps(self, t, r, dt, max_lag):
        t = np.asarray(t)
        r = np.asarray(r) - np.mean(r)

        index = {int(round(ti / dt)): i for i, ti in enumerate(t)}

        max_k = int(max_lag / dt)
        R = np.zeros(max_k + 1)
        Nk = np.zeros(max_k + 1, dtype=int)

        for i, ti in enumerate(t):
            ki = int(round(ti / dt))
            for k in range(max_k + 1):
                j_key = ki + k
                if j_key in index:
                    j = index[j_key]
                    R[k] += r[i] * r[j]
                    Nk[k] += 1

        valid = Nk > 0
        R[valid] /= Nk[valid]
        R[~valid] = np.nan

        tau = np.arange(max_k + 1) * dt
        return tau, R

    def psd_from_autocorr(self, R, dt):
        R = np.nan_to_num(R, nan=0.0)
        S = np.fft.fft(R)
        f = np.fft.fftfreq(len(R), d=dt)

        idx = f >= 0
        return f[idx], np.real(S[idx])

    def cal_toeplitz_cov(
            self, time, res_x, res_y, res_z, dt,
            return_autocovariance=False):
        #
        t_uniform, res_filled_x = self.fill_gaps_with_interp_and_psd(
            np.array(time), res_x, dt
        )
        acov_x = self.estimate_acov_stable(res_filled_x, max_lag=len(res_filled_x) - 1)

        t_uniform, res_filled_y = self.fill_gaps_with_interp_and_psd(
            np.array(time), res_y, dt
        )
        acov_y = self.estimate_acov_stable(res_filled_y, max_lag=len(res_filled_y) - 1)

        t_uniform, res_filled_z = self.fill_gaps_with_interp_and_psd(
            np.array(time), res_z, dt
        )
        acov_z = self.estimate_acov_stable(res_filled_z, max_lag=len(res_filled_z) - 1)

        # '''FFT'''
        eps_x = max(acov_x[0] * 1e-3, 1e-18)
        # Step 1: DCT-I
        S_x = dct(acov_x, type=1, norm=None)

        # Step 2: clipping
        S_clipped_x = np.maximum(S_x, eps_x)
        # Step 3: inverse DCT-I
        acov_corr_x = idct(S_clipped_x, type=1, norm=None)
        # '''FFT'''
        eps_y = max(acov_y[0] * 1e-3, 1e-18)
        S_y = dct(acov_y, type=1, norm=None)
        S_clipped_y = np.maximum(S_y, eps_y)
        acov_corr_y = idct(S_clipped_y, type=1, norm=None)
        # '''FFT'''
        eps_z = max(acov_z[0] * 1e-3, 1e-18)
        S_z = dct(acov_z, type=1, norm=None)
        S_clipped_z = np.maximum(S_z, eps_z)
        acov_corr_z = idct(S_clipped_z, type=1, norm=None)

        if return_autocovariance:
            lag_count = min(
                self.lagNum,
                len(acov_corr_x),
                len(acov_corr_y),
                len(acov_corr_z)
            )
            autocovariance = np.vstack((
                acov_corr_x[:lag_count],
                acov_corr_y[:lag_count],
                acov_corr_z[:lag_count]
            ))
            return autocovariance, float(dt)

        Sigma_x = self.toeplitz_cov_from_acov(acov_corr_x, self.lagNum)
        Sigma_y = self.toeplitz_cov_from_acov(acov_corr_y, self.lagNum)
        Sigma_z = self.toeplitz_cov_from_acov(acov_corr_z, self.lagNum)

        return Sigma_x, Sigma_y, Sigma_z
