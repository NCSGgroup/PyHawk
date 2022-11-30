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


class AdjustOrbit:
    def __init__(self, date_span=Pre_Interface.date_span, arcNo=Pre_Interface.arcNo, sat=Pre_Interface.sat,
                 kind=EnumType.Payload.GNV):
        self.__fr = Frame(EOP().load())
        self.__state_path = None
        self.__state_dir = None
        self.__res_path = None
        self.__res_dir = None
        self.__res_h5 = None
        self.__ode = None
        self.__initTime = None
        self.__initR = None
        self.__endTime = None
        self.__endR = None
        self.__startTime = None
        self.__startPos = None
        self.__startVel = None
        self.__obsGNV = None
        self.__res = None
        self.__ap = None
        self.__ac = None
        self.__secDer = None
        self.__date_span = date_span
        self.__arcNo = arcNo
        self.__sat = sat
        self.__kind = kind

    def configure(self, **kwargs):
        self.__state_path = kwargs.get('state_dir', Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp)
        self.__res_path = kwargs.get('res_dir', Pre_AdjustOrbit.PathOfFiles.OrbitAdjustResTemp)

        """Define the directory to deposit the state vectors"""
        self.__state_dir = self.__state_path.joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        if not self.__state_dir.is_dir():
            self.__state_dir.mkdir(exist_ok=True)
        '''define the directory of res output'''
        self.__res_dir = self.__res_path.joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        if not self.__res_dir.is_dir():
            self.__res_dir.mkdir(exist_ok=True)
        res_filename = self.__res_dir.joinpath(str(self.__arcNo) + '_' + self.__sat.name + '.hdf5')
        self.__res_h5 = h5py.File(res_filename, 'w')
        """step 1: combine the SCA and ACC data to obtain the non-conservative force as well as dadp arc by arc"""
        rd = ArcData(date_span=self.__date_span)
        ap = AccCaliPar(sat=self.__sat, arcNo=self.__arcNo)
        ac = Accelerometer(rd=rd, ap=ap)
        ac.get_and_save()
        self.__ac = ac
        self.__ap = ap
        # """step 2: obtain the GNV orbit of GRACE to initialize the ODE"""
        # GNVA = rd.getData(arc=self.__arcNo, kind=self.__kind, sat=self.__sat)
        # GNVB = rd.getData(arc=self.__arcNo, kind=self.__kind, sat=self.__sat)
        # GPS_Time = GNVA[4, 0]
        # startPosA, startPosB = GNVA[4, 1:4], GNVB[4, 1:4]
        # startVelA, startVelB = GNVA[4, 4:7], GNVB[4, 4:7]
        # startPos = np.vstack((startPosA, startPosB))
        # startVel = np.vstack((startVelA, startVelB))
        """step 3: instance second Derivative"""
        var_2nd_diff = Orbit_var_2nd_diff(self.__fr, rd).setInitData(arc=self.__arcNo, kind=self.__kind, sat=self.__sat)
        self.__secDer = var_2nd_diff
        """step 4: Propagator config"""
        self.__startTime, self.__startPos, self.__startVel = var_2nd_diff.getInitData()
        self.__obsGNV = var_2nd_diff.getInitState()
        self.__ode = GravimetryODE().setInitial(self.__startTime, self.__startPos.copy(), self.__startVel.copy()).set2ndDerivative(var_2nd_diff)
        return self

    def calibrate(self, iteration: int = 2):
        assert iteration >= 2

        '''step 1: orbit integration and save data'''
        ode_res = self.__ode.propagate()
        self.__save_StateVectors(ode_res)

        filename = self.__state_dir.joinpath(str(self.__arcNo) + '_' + self.__sat.name + '.hdf5')
        h5 = h5py.File(filename, 'r')
        self.__initTime = h5['t'][:]
        self.__initR = h5['r'][:]
        h5.close()
        '''step 2: orbit adjustment'''
        orbit_init = np.hstack((self.__startPos, self.__startVel))
        acc_init = np.vstack((self.__ap.getPar(), self.__ap.getPar()))
        '''step 3: start iteration'''
        for i in range(iteration):
            '''make the adjustment'''
            self.__adjust(StateVectors=ode_res[1], isGNV=True)
            orbit_ini, acc_ini = self.__updateIniValue(iniOrbit=orbit_init, iniAcc=acc_init)
            r_ini, v_ini = orbit_ini[:, 0:3], orbit_ini[:, 3:]
            '''step 4: update initial state vector'''
            self.__ode.setInitial(self.__startTime, r_ini, v_ini)
            '''step 5: update accelerometer'''
            self.__ac.updatePar(paralist=acc_ini[0]).get_and_save()
            '''step 6: update force model'''
            self.__secDer.configForce.reConfigure()
            '''step 7: orbit integration once again'''
            ode_res = self.__ode.propagate()
            self.__save_StateVectors(ode_res)

        filename = self.__state_dir.joinpath(str(self.__arcNo) + '_' + self.__sat.name + '.hdf5')
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

    def __save_StateVectors(self, state):
        res_dir = self.__state_dir
        res_filename = res_dir.joinpath(str(self.__arcNo) + '_' + self.__sat.name + '.hdf5')
        h5 = h5py.File(res_filename, 'w')
        h5.create_dataset('t', data=state[0])
        h5.create_dataset('r', data=state[1])
        h5.create_dataset('v', data=state[2])
        h5.close()
        pass

    def draw(self):
        return self.__initTime, self.__initR, self.__endTime, self.__endR

