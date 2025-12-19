import numpy as np

from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.Frame.Frame import Frame
from src.Preference.EnumType import TimeFormat, Payload, SatID
from src.Interface.ArcOperation import ArcData
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_Frame import FrameConfig


class Orbit_2nd_diff(Assemble2ndDerivative):

    def __init__(self, fr: Frame, rd: ArcData):
        super(Orbit_2nd_diff, self).__init__()
        self.__fr = fr
        self.__rd = rd
        self.__initTime = None
        self.__initPos = None
        self.__initVel = None
        self.__PosAndVel = None

    def setInitData(self, arc: int, kind: Payload, sat=SatID.A):
        GNVA = self.__rd.getData(arc=arc, kind=kind, sat=SatID.A)
        GNVB = self.__rd.getData(arc=arc, kind=kind, sat=SatID.B)
        startPosA, startPosB = GNVA[4, 1:4], GNVB[4, 1:4]
        startVelA, startVelB = GNVA[4, 4:7], GNVB[4, 4:7]
        self.__initTime = GNVA[4, 0]
        self.__PosAndVel = np.hstack((GNVA[:, 1:4], GNVB[:, 1:4]))
        self.__initPos = np.vstack((startPosA, startPosB))
        self.__initVel = np.vstack((startVelA, startVelB))
        return self

    def configures(self, arc: int, FMConfig:ForceModelConfig, ParConfig:ParameterConfig, FrameConfig:FrameConfig):
        self._arc = arc
        self.configure(FMConfig=FMConfig, ParameterConfig=ParConfig, FrameConfig=FrameConfig)
        return self

    def secDerivative(self, t, r, v):
        """
        For orbit only in terms of 2nd order differential equation
        :param v: vel vector
        :param t: time
        :param r: position vector
        :return:
        """

        self.__fr = self.__fr.setTime(t, TimeFormat.GPS_second)
        self.setTime(self.__fr).getCS()
        acc = None
        for i in range(len(r)):
            self.setPosAndVel(r[i], v[i]).ChangeSat(SatID.A).calculation()
            if acc is None:
                acc = self.getAcceleration()
            else:
                acc = np.vstack((acc, self.getAcceleration()))

        return acc




