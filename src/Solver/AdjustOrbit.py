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
    def __init__(self, date_span=Pre_Interface.date_span, arcNo=Pre_Interface.arcNo, sat=Pre_Interface.sat):
        self.__fr = Frame(EOP().load())
        self.__state_path = None
        self.__state_dir = None
        self.__ode = None
        self.__initTime = None
        self.__initR = None
        self.__GNV = None
        self.__date_span = date_span
        self.__arcNo = arcNo
        self.__sat = sat
        self.configure()

    def configure(self, **kwargs):
        self.__state_path = kwargs.get('state_dir', Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp)

        """Define the directory to deposit the state vectors"""
        self.__state_dir = self.__state_path.joinpath(self.__date_span[0] + '_' + self.__date_span[1])
        if not self.__state_dir.is_dir():
            self.__state_dir.mkdir(exist_ok=True)
        """step 1: combine the SCA and ACC data to obtain the non-conservative force as well as dadp arc by arc"""
        rd = ArcData(date_span=self.__date_span)
        ap = AccCaliPar(sat=self.__sat, arcNo=self.__arcNo)
        ac = Accelerometer(rd=rd, ap=ap)
        ac.get_and_save()
        """step 2: obtain the GNV orbit of GRACE to initialize the ODE"""
        GNVA = rd.getData(arc=self.__arcNo, kind=EnumType.Payload.GNV, sat=self.__sat)
        GNVB = rd.getData(arc=self.__arcNo, kind=EnumType.Payload.GNV, sat=self.__sat)
        GPS_Time = GNVA[4, 0]
        startPosA, startPosB = GNVA[4, 1:4], GNVB[4, 1:4]
        startVelA, startVelB = GNVA[4, 4:7], GNVB[4, 4:7]
        startPos = np.vstack((startPosA, startPosB))
        startVel = np.vstack((startVelA, startVelB))
        """step 3: instance second Derivative"""
        var_2nd_diff = Orbit_var_2nd_diff(self.__fr)
        """step 4: Propagator config"""
        self.__ode = GravimetryODE().setInitial(GPS_Time, startPos, startVel).set2ndDerivative(var_2nd_diff)

        self.__GNV = np.vstack((GNVA, GNVB))

    def calibrate(self, iteration: int = 3):
        assert iteration >= 2

        '''step 1: orbit integration and save data'''
        ode_res = self.__ode.propagate()
        self.__save_StateVectors(ode_res)

        filename = self.__state_dir.joinpath(str(self.__arcNo) + '_' + self.__sat.name + '.hdf5')
        h5 = h5py.File(filename, 'r')
        self.initTime = h5['t'][:]
        self.initR = h5['r'][:]
        h5.close()
        '''step 2: orbit adjustment'''
        self.__adjust(StateVectors=ode_res[1], isGNV=True)

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
            obs = self.__GNV
        else:
            obs = refOrbit

        '''least-square solver'''
        residual = obs[:, 1:4] - xyz[:, 0:3]

        DesignMatrix = xyz[:, :, 1:]

        b = residual.flatten()
        x = np.linalg.lstsq(DesignMatrix, b, rcond=None)[0]
        # h5.close()
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

