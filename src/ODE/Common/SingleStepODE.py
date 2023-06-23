import numpy as np
from src.Preference.Pre_ODE import ODEConfig


class SingleStepOde:

    def __init__(self):
        self._isRecord = None
        self.__ODEConfig = None
        self.__SingleConfig = None
        self._loadCoeff()

    def _loadCoeff(self):
        # --------------------load the rkn coefficients------------------
        # refer to the Book: "Satellite Orbits: Models, Methods and Applications" Chapter 4
        self._rkn_c = np.array([1 / 10, 1 / 5, 3 / 8, 1 / 2, (7 - np.sqrt(21)) / 14, (7 + np.sqrt(21)) / 14, 1, 1])
        # a0 = np.array([])
        a1 = np.array([1 / 200])
        a2 = np.array([1 / 150, 1 / 75])
        a3 = np.array([171 / 8192, 45 / 4096, 315 / 8192])
        a4 = np.array([5 / 288, 25 / 528, 25 / 672, 16 / 693])
        a5 = np.array([(1003 - 205 * np.sqrt(21)) / 12348, -25 * (751 - 173 * np.sqrt(21)) / 90552,
                       25 * (624 - 137 * np.sqrt(21)) / 43218,
                       -128 * (361 - 79 * np.sqrt(21)) / 237699, (3411 - 745 * np.sqrt(21)) / 24696])
        a6 = np.array([(793 + 187 * np.sqrt(21)) / 12348, -25 * (331 + 113 * np.sqrt(21)) / 90552,
                       25 * (1044 + 247 * np.sqrt(21)) / 43218,
                       -128 * (14885 + 3779 * np.sqrt(21)) / 9745659, (3327 + 797 * np.sqrt(21)) / 24696,
                       -(581 + 127 * np.sqrt(21)) / 1722])
        a7 = np.array([-(157 - 3 * np.sqrt(21)) / 378, 25 * (143 - 10 * np.sqrt(21)) / 2772,
                       -25 * (876 + 55 * np.sqrt(21)) / 3969,
                       1280 * (913 + 18 * np.sqrt(21)) / 596673, -(1353 + 26 * np.sqrt(21)) / 2268,
                       7 * (1777 + 377 * np.sqrt(21)) / 4428, 7 * (5 - np.sqrt(21)) / 36])
        a8 = np.array([1 / 20, 0, 0, 0, 8 / 45, 7 * (7 + np.sqrt(21)) / 360, 7 * (7 - np.sqrt(21)) / 360, 0])

        self._rkn_a = [a1, a2, a3, a4, a5, a6, a7, a8]

        self._rkn_b = np.array(
            [1 / 20, 0, 0, 0, 8 / 45, 7 * (7 + np.sqrt(21)) / 360, 7 * (7 - np.sqrt(21)) / 360, 0, 0])

        self._rkn_bdot = np.array([1 / 20, 0, 0, 0, 16 / 45, 49 / 180, 49 / 180, 1 / 20, 0])

        # -------------------------load the standard RK8 coefficients--------------------------
        # refer to
        self._rk8_c = 1 / 840 * np.array([41, 0, 0, 27, 272, 27, 216, 0, 216, 41])
        self._rk8_a = np.array([4 / 27, 2 / 9, 1 / 3, 1 / 2, 2 / 3, 1 / 6, 1, 5 / 6, 1])
        b1 = np.array([4 / 27])
        b2 = 1 / 18 * np.array([1, 3])
        b3 = 1 / 12 * np.array([1, 0, 3])
        b4 = 1 / 8 * np.array([1, 0, 0, 3])
        b5 = 1 / 54 * np.array([13, 0, -27, 42, 8])
        b6 = 1 / 4320 * np.array([389, 0, -54, 966, -824, 243])
        b7 = 1 / 20 * np.array([-234, 0, 81, -1164, 656, -122, 800])
        b8 = 1 / 288 * np.array([-127, 0, 18, -678, 456, -9, 576, 4])
        b9 = 1 / 820 * np.array([1481, 0, -81, 7104, -3376, 72, -5040, -60, 720])
        self._rk8_sumb = [b1, b2, b3, b4, b5, b6, b7, b8, b9]

        pass

    def configure(self, ODEConfig:ODEConfig):
        self.__ODEConfig = ODEConfig
        self.__SingleConfig = self.__ODEConfig.SingleStep()
        self.__SingleConfig.__dict__.update(self.__ODEConfig.SingleStepConfig.copy())
        return self

    def setInitial(self, r_0, v_0, x_0):
        """
        Initialization
        :param r_0, v_0: initial state vector: pos and velocity
        :param x_0: initial time epoch
        :return:
        """
        self._r_0 = r_0
        self._v_0 = v_0
        self._x_0 = x_0
        return self

    def setRecord(self):
        """
        If isRecord=True, the orbit track (a list of vectors at every epoch) rather than
        the final epoch will be output.
        :param isRecord: Boolean
        :return:
        """
        self._isRecord = self.__SingleConfig.isRecord
        return self

    def setForce(self, func):
        """
        set the right hand of the ODE.
        :param func: for first-oder ODE, func denotes the velocity and force with respect to the derivate of
        position and velocity for second-order ODE, func denotes the only force with respect to the position
        :return:
        """

        self._func = func

        return self

    def propagate(self, Npoints: int, stepsize):
        """
        :param Npoints:
        :param stepsize: Positive--forward integration; Negative--backward integration
        :return: the orbit track (time, pos and vel) at every passed epoch
        """
        assert Npoints > 0
        callfunc = self._callFunc

        x_0 = self._x_0
        r_0 = self._r_0
        v_0 = self._v_0
        X = [x_0]
        R = [r_0]
        V = [v_0]

        for i in range(Npoints):
            x_0, r_0, v_0 = callfunc(self._func, x_0, r_0, v_0, stepsize)
            X.append(x_0)
            R.append(r_0)
            V.append(v_0)

        if self._isRecord:
            return X, R, V
        else:
            return x_0, r_0, v_0

    def _callFunc(self, func, x_0, r_0, v_0, h):
        return x_0, r_0, v_0


'''To Do: fun input'''
class RK4(SingleStepOde):
    def __init__(self):
        super(RK4, self).__init__()

    def _callFunc(self, func, x_0, r_0, v_0, h):

        y_0 = np.append(r_0, v_0)
        k_1 = func(x_0, y_0)
        k_2 = func(x_0 + h / 2, y_0 + (h / 2) * k_1)
        k_3 = func(x_0 + h / 2, y_0 + (h / 2) * k_2)
        k_4 = func(x_0 + h, y_0 + h * k_3)

        y = y_0 + (h / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        x = x_0 + h
        return x, y[0:int(len(y) / 2)], y[int(len(y) / 2):]


class RK8(SingleStepOde):
    def __init__(self):
        super(RK8, self).__init__()

    def _callFunc(self, func, x_0, r_0, v_0, h):
        """
               Test finds that rk8 does not necessarily get higher accuracy than RK4.
               :param func:
               :param x_0:
               :param r_0:
               :param v_0:
               :param h:
               :return:
               The origin is like below
               k_1 = func(x_0, y_0)
               k_2 = func(x_0 + h * (4 / 27), y_0 + (h * 4 / 27) * k_1)
               k_3 = func(x_0 + h * (2 / 9), y_0 + (h / 18) * (k_1 + 3 * k_2))
               k_4 = func(x_0 + h * (1 / 3), y_0 + (h / 12) * (k_1 + 3 * k_3))
               k_5 = func(x_0 + h * (1 / 2), y_0 + (h / 8) * (k_1 + 3 * k_4))
               k_6 = func(x_0 + h * (2 / 3), y_0 + (h / 54) * (13 * k_1 - 27 * k_3 + 42 * k_4 + 8 * k_5))
               k_7 = func(x_0 + h * (1 / 6), y_0 + (h / 4320) * (389 * k_1 - 54 * k_3 + 966 * k_4 - 824 * k_5 + 243 * k_6))
               k_8 = func(x_0 + h, y_0 + (h / 20) * (-234 * k_1 + 81 * k_3 - 1164 * k_4 + 656 * k_5 - 122 * k_6 + 800 * k_7))
               k_9 = func(x_0 + h * (5 / 6),
                          y_0 + (h / 288) * (-127 * k_1 + 18 * k_3 - 678 * k_4 + 456 * k_5 - 9 * k_6 + 576 * k_7 + 4 * k_8))
               k_10 = func(x_0 + h, y_0 + (h / 820) * (
                       1481 * k_1 - 81 * k_3 + 7104 * k_4 - 3376 * k_5 + 72 * k_6 - 5040 * k_7 - 60 * k_8 + 720 * k_9))

               y = y_0 + h / 840 * (41 * k_1 + 27 * k_4 + 272 * k_5 + 27 * k_6 + 216 * k_7 + 216 * k_9 + 41 * k_10)
               """

        a = self._rk8_a
        b = self._rk8_sumb
        c = self._rk8_c

        y_0 = np.hstack((r_0, v_0))

        klist = []
        k_1 = func(x_0, y_0)
        klist.append(k_1)

        # for i in range(9):
        #     r = r_0
        #     v = v_0
        #     for j in range(len(klist)):
        #         r = r + h * klist[j] * b[i][j]
        #         v = v + h * kvlist[j] * b[i][j]
        #
        #     k, kv = func(x_0 + h * a[i], r, v)
        #     klist.append(k)
        #     kvlist.append(kv)

        for i in range(9):
            y = y_0.copy()
            for j in range(len(klist)):
                y += h * klist[j] * b[i][j]

            k = func(x_0 + h * a[i], y)
            klist.append(k)

        y = y_0.copy()
        for i in range(10):
            y = y + h * klist[i] * c[i]

        x = x_0 + h

        return x, y[0:int(len(y) / 2)], y[int(len(y) / 2):]


class RKN(SingleStepOde):
    def __init__(self):
        super(RKN, self).__init__()

    def _callFunc(self, func, x_0, r_0, v_0, h):
        """
          This function follows the coefficients provided by Dormand' Table shown in the book "Orbit" chapter 4
          :param func: the force function should be only relevant to Position rather than Velocity.
          :param x_0: given time epoch
          :param r_0: given position
          :param v_0: given velocity
          :param h: stepsize
          :return: update time, pos and vel
          """
        a = self._rkn_a
        b = self._rkn_b
        bdot = self._rkn_bdot
        c = self._rkn_c

        klist = []
        k_1 = func(x_0, r_0, v_0)
        klist.append(k_1)

        for i in range(7):
            r = r_0 + c[i] * h * v_0
            v = v_0
            for j in range(len(klist)):
                r = r + (h ** 2) * klist[j] * a[i][j]

            k = func(x_0 + h * c[i], r, v)

            klist.append(k)

        r = r_0 + h * v_0
        v = v_0
        for i in range(8):
            r = r + (h ** 2) * klist[i] * b[i]
            v = v + h * klist[i] * bdot[i]

        x = x_0 + h

        return x, r, v

