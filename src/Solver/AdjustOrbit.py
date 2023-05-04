import h5py
import pathlib
import src.Preference.Pre_AdjustOrbit as Pre_AdjustOrbit
import src.Preference.Pre_Interface as Pre_Interface
import src.Preference.Pre_Parameterization as Pre_Parameterization
from src.Interface.ArcOperation import ArcSelect, ArcData
import src.Preference.EnumType as EnumType
import numpy as np
import time
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
from src.ODE.Instance.GravimetryODE import GravimetryODE
from src.Interface.Accelerometer import Accelerometer, AccCaliPar
from src.Frame.Frame import Frame, EOP
import src.Preference.Pre_Accelerometer as Pre_Accelerometer


class AdjustOrbit:
    def __init__(self, date_span=Pre_Interface.date_span, arcNo=Pre_Interface.arcNo, sat=Pre_Interface.sat,
                 kind=EnumType.Payload.GNV):
        # ======================= For demoTest===============================
        self.fisrtR = None
        self.secondR = None
        self.thirdR = None
        # ======================
        self.__fr = Frame(EOP().load())
        self.__ode_res = None
        self.__state_path = None
        self._state_dir = None
        self.__res_path = None
        self._res_dir = None
        self.__res_h5 = None
        self._ode = None
        self.__initTime = None
        self.__initR = None
        self.__endTime = None
        self.__endR = None
        self._startTime = None
        self.__startPos = None
        self.__startVel = None
        self.__obsGNV = None
        self.__res = None
        self.__ap = {}
        self._ac = {}
        self._secDer = None
        self._satName = None
        self._date_span = date_span
        self._arcNo = arcNo
        self._sat = [key for key, value in sat.items() if value]
        self.__kind = kind

    def configure(self, SerDer, **kwargs):
        self.__state_path = kwargs.get('state_dir', Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp)
        self.__res_path = kwargs.get('res_dir', Pre_AdjustOrbit.PathOfFiles.OrbitAdjustResTemp)
        """define satName"""
        for sat in self._sat:
            if self._satName is None:
                self._satName = sat.name
            else:
                self._satName = self._satName + sat.name
        """Define the directory to deposit the state vectors"""
        self._state_dir = self.__state_path.joinpath(self._date_span[0] + '_' + self._date_span[1])
        if not self._state_dir.is_dir():
            self._state_dir.mkdir(exist_ok=True)

        '''define the directory of res output'''
        self._res_dir = self.__res_path.joinpath(self._date_span[0] + '_' + self._date_span[1])
        if not self._res_dir.is_dir():
            self._res_dir.mkdir(exist_ok=True)

        """step 1: combine the SCA and ACC data to obtain the non-conservative force as well as dadp arc by arc"""
        rd = ArcData(date_span=self._date_span)
        for sat in self._sat:
            ap = AccCaliPar(sat=sat, arcNo=self._arcNo)
            ac = Accelerometer(rd=rd, ap=ap)
            ac.get_and_save()
            self._ac[sat] = ac
            self.__ap[sat] = ap

        """step 3: instance second Derivative"""
        self._secDer = SerDer(self.__fr, rd).setInitData(arc=self._arcNo, kind=self.__kind)

        """step 4: Propagator config"""
        self._startTime, self.__startPos, self.__startVel = self._secDer.getInitData()
        self.__obsGNV = self._secDer.getInitState()
        self._ode = GravimetryODE().setInitial(self._startTime, self.__startPos.copy(), self.__startVel.copy()).\
            set2ndDerivative(self._secDer)
        return self

    def calibrate(self, iteration: int = 2):
        assert iteration >= 2
        res_filename = self._res_dir.joinpath(str(self._arcNo) + '_' + self._satName + '.hdf5')
        self.__res_h5 = h5py.File(res_filename, 'w')

        '''step 1: orbit integration and save data'''
        self.__ode_res = self._ode.propagate()
        self._save_StateVectors(self.__ode_res)

        filename = self._state_dir.joinpath(str(self._arcNo) + '_' + self._satName + '.hdf5')
        h5 = h5py.File(filename, 'r')
        self.__initTime = h5['t'][:]
        self.__initR = h5['r'][:]
        h5.close()
        '''step 2: orbit adjustment'''
        orbit_init = np.hstack((self.__startPos, self.__startVel))
        """TODO"""
        acc_init = np.vstack((self.__ap[self._sat[0]].getPar(), self.__ap[self._sat[1]].getPar()))
        '''step 3: start iteration'''
        for i in range(iteration):
            '''make the adjustment'''
            self.__adjust(StateVectors=self.__ode_res[1], isGNV=True)
            orbit_ini, acc_ini = self.__updateIniValue(iniOrbit=orbit_init, iniAcc=acc_init)
            r_ini, v_ini = orbit_ini[:, 0:3], orbit_ini[:, 3:]
            '''step 4: update accelerometer'''
            for j in range(len(self._sat)):
                self._ac[self._sat[j]].updatePar(paralist=acc_ini[j]).get_and_save()
            '''step 5: update force model'''
            self._secDer.reConfigure()
            '''step 6: update initial state vector'''
            self._ode.setInitial(self._startTime, r_ini, v_ini)
            '''step 7: orbit integration once again'''
            self.__ode_res = self._ode.propagate()
            self._save_StateVectors(self.__ode_res)

            if i == 0:
                self.fisrtR = self.__ode_res[1]
            elif i == 1:
                self.secondR = self.__ode_res[1]
            elif i == 2:
                self.thirdR = self.__ode_res[1]

        filename = self._state_dir.joinpath(str(self._arcNo) + '_' + self._satName + '.hdf5')
        h5 = h5py.File(filename, 'r')
        self.__endTime = h5['t'][:]
        self.__endR = h5['r'][:]
        h5.close()

        return self

    def __updateIniValue(self, iniOrbit: np.ndarray, iniAcc: np.ndarray):
        res = self.__res
        h5 = self.__res_h5

        key = Pre_Parameterization.TransitionMatrix
        if key.isRequired:
            iniOrbit += res[:, 0:key.Parameter_Number]
            if key.__name__ in h5:
                h5[key.__name__][()] = iniOrbit
            else:
                h5.create_dataset(key.__name__, data=iniOrbit)

        key = Pre_Parameterization.Accelerometer
        if key.isRequired:
            left = Pre_Parameterization.TransitionMatrix.Parameter_Number
            right = Pre_Parameterization.TransitionMatrix.Parameter_Number + key.Parameter_Number
            iniAcc += res[:, left:right]
            if key.__name__ in h5:
                h5[key.__name__][()] = iniAcc
            else:
                h5.create_dataset(key.__name__, data=iniAcc)
        return iniOrbit, iniAcc

    def __adjust(self, StateVectors: np.ndarray, isGNV: bool, refOrbit: np.ndarray = None):
        """
        :param isGNV: is the reference orbit GNV ?
        :param StateVectors: orbit integration output
        :param refOrbit: reference orbit. It could be the GNV orbit or the other orbit like kinematic orbit transferred
        from outside; The first column must be the time stamp, and the followings are x,y,z, and optionally vx,vy,vz.
        :return:
        """
        xyz = StateVectors

        obs = None
        if isGNV:
            obs = self.__obsGNV
        else:
            obs = refOrbit
        '''least-square solver'''
        residual = obs - xyz[:, :, 0]

        DesignMatrixA = xyz[:, 0:3, 1:].reshape((len(xyz) * 3, -1))
        bA = residual[:, 0:3].flatten()
        xA = np.linalg.lstsq(DesignMatrixA, bA, rcond=None)[0]

        DesignMatrixB = xyz[:, 3:6, 1:].reshape((len(xyz) * 3, -1))
        bB = residual[:, 3:6].flatten()
        xB = np.linalg.lstsq(DesignMatrixB, bB, rcond=None)[0]

        self.__res = np.vstack((xA, xB))
        return self

    def _save_StateVectors(self, state):
        res_dir = self._state_dir
        res_filename = res_dir.joinpath(str(self._arcNo) + '_' + self._satName + '.hdf5')
        h5 = h5py.File(res_filename, 'w')
        h5.create_dataset('t', data=state[0])
        h5.create_dataset('r', data=state[1])
        h5.create_dataset('v', data=state[2])
        h5.close()
        pass

    def draw(self):
        return self.__initTime, self.__initR, self.__endTime, self.__endR

    def getR(self):
        return self.fisrtR, self.secondR, self.thirdR


class AdjustOrbitForDM(AdjustOrbit):
    """
    Make an orbit integration with the calibrated parameters again. No calibration is required.
    Configuration could be totally different anyway from the orbit calibration.
    """
    def __init__(self, date_span=Pre_Interface.date_span, arcNo=Pre_Interface.arcNo, sat=Pre_Interface.sat,
                 kind=EnumType.Payload.GNV):
        super().__init__(date_span, arcNo, sat, kind)
        self._arcNo = arcNo

    def updateInit(self):
        sat = self._sat
        ac = self._ac
        time_ini = self._startTime

        cali_res = self.getInit()
        if EnumType.ParaType.Accelerometer.name in cali_res.keys():
            acc_par = cali_res[EnumType.ParaType.Accelerometer.name]
            scale = acc_par[0]
            bias = acc_par[1]
            for j in range(len(sat)):
                ac[sat[j]].setParFromList(paralist=[scale[j], bias[j]]).get_and_save()
            self._secDer.reConfigure()  # reload the accelerometers from files.

        if EnumType.ParaType.TransitionMatrix.name in cali_res.keys():
            a = cali_res[EnumType.ParaType.TransitionMatrix.name]
            r_ini, v_ini = a[:, 0:3], a[:, 3:]
            self._ode.setInitial(time_ini, r_ini, v_ini)

        return self

    def getInit(self, **kwargs):
        res_filename = self._res_dir.joinpath(str(self._arcNo) + '_' + self._satName + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        res = {}
        for key in h5:
            res[key] = h5[key][()]
        h5.close()
        # --------------------------------------------------------------
        '''pass the parameters into accelerometers'''
        acc_strategy = kwargs.get("strategy", Pre_Accelerometer.cali)
        isScale = acc_strategy['Scale']
        isBias_constant = acc_strategy['Bias_constant']
        isBias_trend = acc_strategy['Bias_trend']
        isBias_quadratic = acc_strategy['Bias_quadratic']
        isCali = [isScale, isBias_constant, isBias_trend, isBias_quadratic]
        cali = []
        for term in isCali:
            '''span to 1-d list'''
            cali = cali + term
        cali = np.vstack((cali, cali))

        cali_value = kwargs.get("strategy", Pre_Accelerometer.cali_value)
        scale = cali_value['Scale']
        bias_constant = cali_value['Bias_constant']
        bias_trend = cali_value['Bias_trend']
        bias_quadratic = cali_value['Bias_quadratic']
        scale = np.array(scale, dtype=np.float).reshape((3, 3))
        bias = np.array([bias_constant, bias_trend, bias_quadratic], dtype=np.float)
        scale, bias = scale.flatten(), bias.flatten()
        scale, bias = np.vstack((scale, scale)), np.vstack((bias, bias))

        if EnumType.ParaType.Accelerometer.name in res.keys():
            accIni = res[EnumType.ParaType.Accelerometer.name]
            valuelist = np.zeros((2, 18), dtype=np.float)
            valuelist[cali] = np.array(accIni.flatten())

            '''replace'''
            scale[cali[:, 0:9]] = valuelist[:, 0:9][cali[:, 0:9]]
            bias[cali[:, 9:]] = valuelist[:, 9:][cali[:, 9:]]
            res[EnumType.ParaType.Accelerometer.name] = [scale, bias]

        return res

    def calibrate(self,**kwargs):
        ode = self._ode
        '''step 15: orbit integration'''
        if 'ChangeTempPath' in kwargs.keys():
            self._state_dir = kwargs['ChangeTempPath']
        # ode.setRecord(dataName=date_begin + '_' + date_end + '_' + str(arcNo) + '_' + sat.name)
        ode_res = ode.propagate()
        self._save_StateVectors(ode_res)
        pass

