import numpy as np
from src.Preference.EnumType import RKOrder
from src.Preference.Pre_ODE import RK


class RungeKutta:
    """
    This is a single-step orbit integrator, which is usually used for the self-starting of multiple-step integrator
    like Storm-Cowell (to develop), GaussJackson (provided) and ABM8 (provided).
    The standard Runge-kutta is tailored to the problem of "first order differential equation (ODE)": RK4 and RK8.
    However, the Runge-kutta-nystorm methods are tailored to the problem of "second order differential equation (ode)": RKN-8
    Reference:
    1. Book: Satellite Orbits: Models, Methods and Applications Chapter 4
    2. https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.integrate.ode.html
    3. https://ww2.mathworks.cn/matlabcentral/fileexchange/55533-satellite-orbits-models-methods-and-applications
    4. https://ww2.mathworks.cn/matlabcentral/fileexchange/?term=authorid%3A1018620

    Note: all the integrations are done with GCRS frame.
    Suggestion: Experiments find the RKN has the highest accuracy among all the methods,
    and the Python embedded ODE module is the second. Therefore, the RKN should be the first option.
    (Run the demo to check the results)

    Notice: when you want an integration backward, the only thing needed to do is to set the STEPSIZE in a negative
    number! And don't change the force!
    """

    def __init__(self, order=RK.RKOrder):
        self.__order = order
        self.__loadCoeff()
        self.setRecord()

    def __loadCoeff(self):
        # --------------------load the rkn coefficients------------------
        # refer to the Book: "Satellite Orbits: Models, Methods and Applications" Chapter 4
        self.__rkn_c = np.array([1 / 10, 1 / 5, 3 / 8, 1 / 2, (7 - np.sqrt(21)) / 14, (7 + np.sqrt(21)) / 14, 1, 1])
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

        self.__rkn_a = [a1, a2, a3, a4, a5, a6, a7, a8]

        self.__rkn_b = np.array(
            [1 / 20, 0, 0, 0, 8 / 45, 7 * (7 + np.sqrt(21)) / 360, 7 * (7 - np.sqrt(21)) / 360, 0, 0])

        self.__rkn_bdot = np.array([1 / 20, 0, 0, 0, 16 / 45, 49 / 180, 49 / 180, 1 / 20, 0])

        # -------------------------load the standard RK8 coefficients--------------------------
        # refer to
        self.__rk8_c = 1 / 840 * np.array([41, 0, 0, 27, 272, 27, 216, 0, 216, 41])
        self.__rk8_a = np.array([4 / 27, 2 / 9, 1 / 3, 1 / 2, 2 / 3, 1 / 6, 1, 5 / 6, 1])
        b1 = np.array([4 / 27])
        b2 = 1 / 18 * np.array([1, 3])
        b3 = 1 / 12 * np.array([1, 0, 3])
        b4 = 1 / 8 * np.array([1, 0, 0, 3])
        b5 = 1 / 54 * np.array([13, 0, -27, 42, 8])
        b6 = 1 / 4320 * np.array([389, 0, -54, 966, -824, 243])
        b7 = 1 / 20 * np.array([-234, 0, 81, -1164, 656, -122, 800])
        b8 = 1 / 288 * np.array([-127, 0, 18, -678, 456, -9, 576, 4])
        b9 = 1 / 820 * np.array([1481, 0, -81, 7104, -3376, 72, -5040, -60, 720])
        self.__rk8_sumb = [b1, b2, b3, b4, b5, b6, b7, b8, b9]

        pass

    def setRecord(self, isRecord=RK.isRecord):
        """
        If isRecord=True, the orbit track (a list of vectors at every epoch) rather than
        the final epoch will be output.
        :param isRecord: Boolean
        :return:
        """
        self.__isRecord = isRecord
        return self

    def setInitial(self, r_0, v_0, x_0):
        """
        Initialization
        :param r_0, v_0: initial state vector: pos and velocity
        :param x_0: initial time epoch
        :return:
        """
        self.__r_0 = r_0
        self.__v_0 = v_0
        self.__x_0 = x_0
        return self

    def setForce(self, func):
        """
        set the right hand of the ODE.
        :param func: for first-oder ODE, func denotes the velocity and force with respect to the derivate of
        position and velocity for second-order ODE, func denotes the only force with respect to the position
        :return:
        """

        self.__func = func

        return self

    def propagate(self, Npoints: int = RK.Npoints, stepsize=RK.stepsize):
        """

        :param Npoints:
        :param stepsize: Positive--forward integration; Negative--backward integration
        :return: the orbit track (time, pos and vel) at every passed epoch
        """
        assert Npoints > 0
        callfunc = None
        if self.__order.name == 'rk4':
            callfunc = self.__rk4_standard
        elif self.__order.name == 'rk8':
            callfunc = self.__rk8_standard
        elif self.__order.name == 'rkn':
            callfunc = self.__rkn7_Dormand

        x_0 = self.__x_0
        r_0 = self.__r_0
        v_0 = self.__v_0
        X = [x_0]
        R = [r_0]
        V = [v_0]
        for i in range(Npoints):
            x_0, r_0, v_0 = callfunc(self.__func, x_0, r_0, v_0, stepsize)
            X.append(x_0)
            R.append(r_0)
            V.append(v_0)

        if self.__isRecord:
            return X, R, V
        else:
            return x_0, r_0, v_0

    def __rk4_standard(self, func, x_0, r_0, v_0, h):

        y_0 = np.append(r_0, v_0)
        k_1 = func(x_0, y_0)
        k_2 = func(x_0 + h / 2, y_0 + (h / 2) * k_1)
        k_3 = func(x_0 + h / 2, y_0 + (h / 2) * k_2)
        k_4 = func(x_0 + h, y_0 + h * k_3)

        y = y_0 + (h / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        x = x_0 + h
        return x, y[0:int(len(y) / 2)], y[int(len(y) / 2):]

    def __rk8_standard(self, func, x_0, r_0, v_0, h):
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

        a = self.__rk8_a
        b = self.__rk8_sumb
        c = self.__rk8_c

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

    def __rkn7_Dormand(self, func, x_0, r_0, v_0, h):
        """
        This function follows the coefficients provided by Dormand' Table shown in the book "Orbit" chapter 4
        :param func: the force function should be only relevant to Position rather than Velocity.
        :param x_0: given time epoch
        :param r_0: given position
        :param v_0: given velocity
        :param h: stepsize
        :return: update time, pos and vel
        """
        a = self.__rkn_a
        b = self.__rkn_b
        bdot = self.__rkn_bdot
        c = self.__rkn_c

        klist = []
        # print(x_0)
        # print(r_0)
        # print(v_0)
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

    # def __rkn7_doubleIntegration(self, func1, func2, x_I, r_I, v_I, x_II, r_II, v_II, h):
    #     """
    #     This is an extension of the rkn7, as it can integrate two sets of variables at the same time.
    #     The function may get benefit from a sharing evaluation of the force model that usually costs much time.
    #
    #     In practise, this function would be called to initialize the GaussJackson Integrator for ORBIT AND VARIATION
    #     EQUATIONS.
    #
    #     TO BE DEVELOPED:
    #     As it can be seen, this function could be also extended to deal with more sets of variables (>=2)
    #
    #     :param func1: force model for set 1
    #     :param func2: force model for set 2
    #         set 1
    #     :param x_I:
    #     :param r_I:
    #     :param v_I:
    #         set 2
    #     :param Ix_0:
    #     :param Ir_0:
    #     :param Iv_0:
    #
    #     :param h: step size
    #     :return:
    #     """
    #
    #     a = self.__rkn_a
    #     b = self.__rkn_b
    #     bdot = self.__rkn_bdot
    #     c = self.__rkn_c
    #
    #     klist_I = []
    #     k_I = func1(x_I, r_I)
    #     klist_I.append(k_I)
    #
    #     klist_II = []
    #     k_II = func2(x_II, r_II)
    #     klist_II.append(k_II)
    #
    #     for i in range(7):
    #         R_I = r_I + c[i] * h * v_I
    #
    #         R_II = r_II + c[i] * h * v_II
    #
    #         for j in range(len(klist_I)):
    #             R_I += (h ** 2) * klist_I[j] * a[i][j]
    #
    #             R_II += (h ** 2) * klist_II[j] * a[i][j]
    #
    #         K_I = func1(x_I + h * c[i], R_I)
    #         K_II = func2(x_II + h * c[i], R_II)
    #
    #         klist_I.append(K_I)
    #         klist_II.append(K_II)
    #
    #     R_I = r_I + h * v_I
    #     V_I = v_I
    #
    #     R_II = r_II + h * v_II
    #     V_II = v_II
    #
    #     for i in range(8):
    #         R_I += (h ** 2) * klist_I[i] * b[i]
    #         V_I += h * klist_I[i] * bdot[i]
    #
    #         R_II += (h ** 2) * klist_II[i] * b[i]
    #         V_II += h * klist_II[i] * bdot[i]
    #
    #     X_I = x_I + h
    #     X_II = x_II + h
    #
    #     return X_I, R_I, V_I, X_II, R_II, V_II
    #
    # def propagate_doubleIntegration(self, Npoints: int, stepsize):
    #
    #     callfunc = self.__rkn7_doubleIntegration
    #
    #     x_0 = self.__x_0
    #     r_0 = self.__r_0
    #     v_0 = self.__v_0
    #
    #     rr_0 = np.array([0.9, 0, 0])
    #     vv_0 = np.array([0, 1.105541596785133, 0])
    #     xx_0 = 0
    #
    #     for i in range(Npoints):
    #         x_0, r_0, v_0, xx_0, rr_0, vv_0 = \
    #             callfunc(self.__func, self.__func, x_0, r_0, v_0, xx_0, rr_0, vv_0, stepsize)
    #
    #     return x_0, r_0, v_0, xx_0, rr_0, vv_0

