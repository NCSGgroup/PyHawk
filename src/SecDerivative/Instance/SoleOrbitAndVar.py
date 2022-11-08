from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.Frame.Frame import Frame
from src.Preference.EnumType import TimeFormat
from src.SecDerivative.VarEqn import VarEqn
from src.SecDerivative.MatrixVarEqn import MatrixVarEqn


class Orbit_var_2nd_diff(Assemble2ndDerivative):

    def __init__(self, fr: Frame):
        self.__fr = fr
        super(Orbit_var_2nd_diff, self).__init__()

    def secDerivative(self, t, PhirSr, PhivSv):
        """
        For orbit and variation equations in terms of 2nd order differential equation
        :param t: time
        :param PhirSr: StateVec [3*1] + Transition Matrix  [3*6]
                       + Parameter Sensitivity matrix  [3*np] => one dimension
        :param PhivSv: time derivative of the PhirSr
        :return:
        """
        r, v = VarEqn.getRVfromVarEq2ndOrder(PhirSr, PhivSv)

        # self.setPar(t, r, v)
        # acc = self.get_acceleration()
        # ar = self.get_du2xyz()
        # ap = self.get_du2p()

        self.__fr = self.__fr.setTime(t, TimeFormat.GPS_second)
        self.setPosAndVel(r, v).setTime(self.__fr)

        acc = self.getAcceleration()
        ar = self.getDaDx()
        ap = self.getDaDp()

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
        if ar is None:
            bool1 = False
        if ap is None:
            bool2 = False

        '''calculate the second order differentials'''
        diff2nd = VarEqn.VarEq2ndOrder(PhirSr, PhivSv, ar, ap, acc,
                                       isIncludeStateVec=True,
                                       isIncludeTransitionMatrix=bool1,
                                       isIncludeSensitivityMatrix=bool2)
        return diff2nd


class MatrixOrbit_var_2nd_diff(Assemble2ndDerivative):

    def __init__(self, fr: Frame):
        self.__fr = fr
        super(MatrixOrbit_var_2nd_diff, self).__init__()

    def secDerivative(self, t, PhirSr, PhivSv):
        """
        For orbit and variation equations in terms of 2nd order differential equation
        :param t: time
        :param PhirSr: StateVec [3*1] + Transition Matrix  [3*6]
                       + Parameter Sensitivity matrix  [3*np] => one dimension
        :param PhivSv: time derivative of the PhirSr
        :return:
        """
        r, v = MatrixVarEqn.getRVfromVarEq2ndOrder(PhirSr, PhivSv)
        # self.setPar(t, r, v)
        # acc = self.get_acceleration()
        # ar = self.get_du2xyz()
        # ap = self.get_du2p()

        self.__fr = self.__fr.setTime(t, TimeFormat.GPS_second)
        self.setPosAndVel(r, v).setTime(self.__fr)

        acc = self.getAcceleration()
        ar = self.getDaDx()
        ap = self.getDaDp()

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
        if ar is None:
            bool1 = False
        if ap is None:
            bool2 = False

        '''calculate the second order differentials'''
        diff2nd = MatrixVarEqn.VarEq2ndOrder(PhirSr, PhivSv, ar, ap, acc,
                                       isIncludeStateVec=True,
                                       isIncludeTransitionMatrix=bool1,
                                       isIncludeSensitivityMatrix=bool2)
        return diff2nd