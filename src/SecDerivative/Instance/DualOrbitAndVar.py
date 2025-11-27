import numpy as np
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.Frame.Frame import Frame
from src.Preference.EnumType import TimeFormat, Payload, SatID
from src.SecDerivative.VarEqn import VarEqn
from src.SecDerivative.MatrixVarEqn import MatrixVarEqn
from src.Interface.ArcOperation import ArcData
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_Frame import FrameConfig
from src.Interface.Accelerometer import Accelerometer, Accelerometer_V2, AccCaliPar, Accelerometer_V3, AccCaliPar_V3


class Orbit_var_2nd_diff(Assemble2ndDerivative):

    def __init__(self, fr: Frame, rd: ArcData):
        super(Orbit_var_2nd_diff, self).__init__()
        self.__fr = fr
        self.__rd = rd
        self.Time = None
        self.__ArcLen = None
        self.__initTime = None
        self.__initPos = None
        self.__initVel = None
        self.__PosAndVel = None
        self.__commonTime = None
        self.node = None

        self.accA = []
        self.accB = []

    def configures(self, arc: int, FMConfig:ForceModelConfig, ParConfig:ParameterConfig, FrameConfig:FrameConfig):
        self._arc = arc
        self._ParameterConfig = ParConfig
        '''config AccelerometerConfig'''
        self._AccelerometerConfig = self._ParameterConfig.Accelerometer()
        self._AccelerometerConfig.__dict__.update(self._ParameterConfig.AccelerometerConfig.copy())
        self.adjustlength_x = self._AccelerometerConfig.X_AdjustLength * 3600
        self.adjustlength_y = self._AccelerometerConfig.Y_AdjustLength * 3600
        self.adjustlength_z = self._AccelerometerConfig.Z_AdjustLength * 3600

        '''init acc'''

        self.configure(FMConfig=FMConfig, ParameterConfig=ParConfig, FrameConfig=FrameConfig)
        return self

    def setInitData(self, kind: Payload):
        GNVA = self.__rd.getData(arc=self._arc, kind=kind, sat=SatID.A)
        GNVB = self.__rd.getData(arc=self._arc, kind=kind, sat=SatID.B)

        time_A = GNVA[:, 0]
        time_B = GNVB[:, 0]

        if len(time_A) != len(time_B):
            self.__ArcLen = len(time_B) if len(time_A) > len(time_B) else len(time_A)
            self.Time = time_B if len(time_A) > len(time_B) else time_A
        else:
            self.__ArcLen = len(time_B)
            self.Time = time_B
            # self.__ArcLen = len(time_A)
            commonTime = set(time_A.astype(np.int64))
            commonTime = commonTime & set(time_B.astype(np.int64))
            commonTime = np.array(list(commonTime))
            commonTime.sort()
            index1 = [list(time_A).index(x) for x in commonTime]
            index2 = [list(time_B).index(x) for x in commonTime]
            GNVA = GNVA[index1, :]
            GNVB = GNVB[index2, :]

        '''divide x layer'''
        self.node_x = self.divideLayer(self.adjustlength_x)
        '''divide y layer'''
        self.node_y = self.divideLayer(self.adjustlength_y)
        '''divide z layer'''
        self.node_z = self.divideLayer(self.adjustlength_z)

        self.node = (len(self.node_x) + len(self.node_y) + len(self.node_z) - 3) * 2

        self.__initTime = GNVA[4, 0]
        startPosA, startPosB = GNVA[4, 1:4], GNVB[4, 1:4]
        startVelA, startVelB = GNVA[4, 4:7], GNVB[4, 4:7]

        # self.__commonTime = commonTime
        self.__PosAndVel = np.hstack((GNVA[:, 1:4], GNVB[:, 1:4]))
        self.__initPos = np.vstack((startPosA, startPosB))
        self.__initVel = np.vstack((startVelA, startVelB))
        return self

    def getInitData(self):
        return self.__initTime, self.__initPos, self.__initVel

    def getCommonTime(self):
        return self.__commonTime

    def getInitState(self):
        return self.__PosAndVel

    def secDerivative(self, t, PhirSr, PhivSv):
        """
        For orbit and variation equations in terms of 2nd order differential equation
        :param t: time
        :param PhirSr: StateVec [3*1] + Transition Matrix  [3*6]
                       + Parameter Sensitivity matrix  [3*np] => one dimension
        :param PhivSv: time derivative of the PhirSr
        :return:
        """
        rA, vA = MatrixVarEqn.getRVfromVarEq2ndOrder(PhirSr[0:3], PhivSv[0:3])
        rB, vB = MatrixVarEqn.getRVfromVarEq2ndOrder(PhirSr[3:6], PhivSv[3:6])

        # self.setPar(t, r, v)
        # acc = self.get_acceleration()
        # ar = self.get_du2xyz()
        # ap = self.get_du2p()

        self.__fr = self.__fr.setTime(t, TimeFormat.GPS_second)
        self.setTime(self.__fr).getCS()

        self.setPosAndVel(rA, vA).ChangeSat(SatID.A).calculation()
        accA = self.getAcceleration()
        arA = self.getDaDx()
        apA = self.getDaDp()

        self.setPosAndVel(rB, vB).ChangeSat(SatID.B).calculation()
        accB = self.getAcceleration()
        arB = self.getDaDx()
        apB = self.getDaDp()

        '''benchmark acc'''

        self.accA.append(accA)
        self.accB.append(accB)
        '''Computation efficiency test'''
        '''Test Result: MOST COST TIME:'''
        '''1. Ocean Tide, 2. EOP correction, 3. Planets, 4.  '''
        # begin = Ti.time()
        #
        # for i in range(1000):
        #     self.setTime(t)
        #     # print('1 Cost time: %s ms' % ((Ti.time() - begin) * 1000))
        #
        #     # begin = Ti.time()
        #     self.setSATinfo(self.__sat, r, v)
        #     # print('2 Cost time: %s ms' % ((Ti.time() - begin) * 1000))
        #
        #     # begin = Ti.time()
        #     acc = self.get_acceleration()
        #     ar = self.get_du2xyz()
        #     ap = self.get_du2p()
        #
        # print('3 Cost time: %s ms' % ((Ti.time() - begin) * 1000))0

        bool1, bool2 = True, True
        if arA is None:
            bool1 = False
        if apA is None:
            bool2 = False

        '''calculate the second order differentials'''
        diff2ndA = MatrixVarEqn.VarEq2ndOrder(PhirSr[0:3], PhivSv[0:3], arA, apA, accA,
                                       isIncludeStateVec=True,
                                       isIncludeTransitionMatrix=bool1,
                                       isIncludeSensitivityMatrix=bool2)

        diff2ndB = MatrixVarEqn.VarEq2ndOrder(PhirSr[3:6], PhivSv[3:6], arB, apB, accB,
                                       isIncludeStateVec=True,
                                       isIncludeTransitionMatrix=bool1,
                                       isIncludeSensitivityMatrix=bool2)

        return np.vstack((diff2ndA, diff2ndB))

    def getArcLen(self):
        return self.__ArcLen

    def getacc(self):
        return self.accA, self.accB

    def divideLayer(self, adjustlength):
        node = [self.Time[0]]
        t1 = self.Time[0]
        while True:
            t2 = t1 + adjustlength
            if t2 - self.Time[-1] <= 0:
                '''difference less than 0.5 hour, end'''
                node.append(t2)
                t1 = t2
            elif len(self.Time[self.Time >= t2]) <= 100:
                if len(self.Time[self.Time >= t2]) == 0:
                    node.append(self.Time[-1] + 5)
                else:
                    '''the rest points are too short, e.g., less than 100: include the rest into the last arc'''
                    node[-1] = self.Time[-1] + 5  # +10, a little bigger than the last point to ensure all is included
                break

        return node

    def getTimes(self):
        return self.node