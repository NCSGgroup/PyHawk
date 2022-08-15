import numpy as np
from src.ODE.ODEcoeffGenerator import ODEcoeffGenerator
from src.Preference.Pre_ODE import GJ


class GaussJackson:
    """
    Linear multi-step methods are used for the numerical solution of ordinary differential equations.
    Conceptually, a numerical method starts from an initial point and then takes a short step forward in time
    to find the next solution point. The process continues with subsequent steps to map out the solution.
    Single-step methods (such as Euler's method) refer to only one previous point and its derivative to determine
    the current value. Methods such as Runge–Kutta take some intermediate steps (for example, a half-step)
    to obtain a higher order method, but then discard all previous information before taking a second step.
    Multi-step methods attempt to gain efficiency by keeping and using the information from previous steps
    rather than discarding it. Consequently, multi-step methods refer to several previous points and derivative values.
    In the case of linear multi-step methods, a linear combination of the previous points and derivative values is used.

    This program is a module which performs to do orbit integral using GJ method.

    The reference is
    1. "M. Berry, Healy, L. "Implementation of Gauss-Jackson Integration for
    Orbit Propagation". The Journal of Astronautical Sciences. V 52. N 3. Sept. 2004. pp 331-357.".
    2. Book: Satellite Orbits: Models, Methods and Applications, Chapter 4
    3. https://ww2.mathworks.cn/matlabcentral/fileexchange/36542-gauss-jackson-eighth-order-ode-solver-fixed-step-size

    !Note: for the best, the initial epochs (8th order as an example) are suggested to set to
    like -4, -3,-2, -1 0 1 2 3 4, which means several epochs
    before the zero epoch "the true starting points" should be given.
    !Note: Integral must be done in an inertial coordinate frame, like the GCRS frame.
    """

    def __init__(self, order: int = GJ.order):
        '''
        Define the basic attributions of the propagator
        :param order: the propagator should be preferred with an order less than 18 (to lower computation burden) and
        greater than 4 (to gain computation precision)
        '''
        self.__order = order
        assert np.mod(order, 2) == 0 and 4 <= order <= 18

        # default setting
        self.setIterArgs()

        self.setRecord()

        self.__Adam_Coef, self.__GJ_Coeff = self.__generateCoef(order)

        pass

    def setIterArgs(self, iterNum: int = GJ.iterNum, rltot=GJ.rltot):
        """
        :param rltot: tolerance of stopping an iteration.
        :param iterNum: define a maximal number of iteration to lower computation burden.
        :return:
        """
        self.__iterNum = iterNum
        self.__rltot = rltot
        return self

    def setRecord(self, isRecord=GJ.isRecord):
        """
        If isRecord=True, the orbit track (a list of vectors at every epoch) rather than
        the final epoch will be output.
        :param isRecord: Boolean
        :return:
        """
        self.__isRecord = isRecord
        return self

    def __generateCoef(self, order: int):
        '''
        Generate or load the coefficients of propagator at given order
        :param order:
        :return: Adams coefficients and GJ coefficients
        '''
        dim = (order + 2, order + 1)
        ODE = ODEcoeffGenerator(order)
        Adam_Coef = np.array(ODE.SummedAdam_ordinate).astype(np.float)
        GJ_Coeff = np.array(ODE.GaussJackson_ordinate).astype(np.float)

        assert Adam_Coef.shape == dim and GJ_Coeff.shape == dim

        return Adam_Coef, GJ_Coeff

    def setInitial(self, iniX: list, iniR: list, iniV: list):
        """
        Define the initial value
        Note, the iniX ane iniY should have the same size of "order+1", otherwise an error report will be returned.
        :param iniX: initial time epochs of given time series
        :param iniR: r (position) of given time series.
        :param iniV: v (velocity) of given time series
        :return:
        """

        self.__iniX = iniX
        self.__iniR = iniR
        self.__iniV = iniV

        # make sure the initial points have a number of "order+1"
        assert len(iniX) == len(iniR) == len(iniV) == (self.__order + 1)

        return self

    def setForce(self, func):
        """
        Force function defined outside
        :param func: force function
        :return:
        """
        self.__func = func
        return self

    def propagate(self, N: int = GJ.Npoints, hs=GJ.stepsize):
        """
        The propagator starts to work for N steps from the given initial epoch
        :param N: Steps the propagator will extend
        :param hs: step size
        :return:
        """

        assert N > 0
        OrbitX, OrbitR, OrbitV = [], [], []

        X, R, V = np.array(self.__iniX), np.array(self.__iniR), np.array(self.__iniV)
        acc = self.__evaluation(X, R, V)
        isConverge = False
        iter = 0
        sn = 0
        Sn = 0

        '''start-up: mid-correction'''
        while (not isConverge) and iter < 10:
            iter += 1
            X, R, V, acc, sn, Sn, isConverge = self.__startup(X, R, V, acc, hs, self.__rltot)


        '''Main part'''
        for i in range(N):
            '''# repeated PE(CECECE...) procedures'''
            '''predictor'''
            X, R, V, acc, Sn, sn = self.__predictor(X, R, V, acc, sn, Sn, hs)

            isConverge = False

            iter = 0
            while (not isConverge) and iter < self.__iterNum:
                '''# Corrector'''
                iter += 1
                R, V, acc, isConverge = self.__corrector(X, R, V, acc, sn, Sn, hs, self.__rltot)
                pass

            if self.__isRecord:
                OrbitX.append(X[-1])
                OrbitR.append(R[-1])
                OrbitV.append(V[-1])

        if self.__isRecord:
            return OrbitX, OrbitR, OrbitV
        else:
            return X[-1], R[-1], V[-1]

    def propagate_one_step(self, hs, isFirstStep: True, dataIn):
        """
        One step forward once the function is called
        :param hs:
        :param isFirstStep: Mid-correction would be done if this is the first step of the whole integration
        :return:
        """

        # start-up
        if isFirstStep:

            X, R, V = np.array(self.__iniX), np.array(self.__iniR), np.array(self.__iniV)
            acc = self.__evaluation(X, R, V)
            isConverge = False
            iter = 0
            sn = 0
            Sn = 0
            while (not isConverge) and iter < 10:
                iter += 1
                X, R, V, acc, sn, Sn, isConverge = self.__startup(X, R, V, acc, hs, self.__rltot)
            return X, R, V, acc, sn, Sn
        else:
            (X, R, V, acc, sn, Sn) = dataIn

        # predictor
        X, R, V, acc, Sn, sn = self.__predictor(X, R, V, acc, sn, Sn, hs)

        isConverge = False

        iter = 0
        while (not isConverge) and iter < self.__iterNum:
            # Corrector
            iter += 1
            R, V, acc, isConverge = self.__corrector(X, R, V, acc, sn, Sn, hs, self.__rltot)

        return X, R, V, acc, sn, Sn

    @property
    def getOrder(self):
        return self.__order

    def __startup(self, X: np.ndarray, R: np.ndarray, V: np.ndarray
                  , acc: np.ndarray, hs, RelTol):
        """
        See ref 1, the step 3 of start-up
        :param X: Time epoch
        :param R: pos
        :param V: velocity (mostly no importance)
        :param acc: accelerations
        :param hs: stepsize
        :param RelTol: tolerance of the iteration
        :return: updated pos and vel after converge test
        """

        index = self.__get_index
        order = self.__order
        b = self.__Adam_Coef
        a = self.__GJ_Coeff

        dim = np.shape(X)[0]
        npar = np.shape(R)[1]
        shape = (dim, npar)

        assert np.shape(R) == np.shape(V) == np.shape(acc) == shape
        assert dim == order + 1

        # calculate C1 by Eq. 73 in ref 1
        # print(V[index(0), :])
        # print(V[index(0)])

        C1 = V[index(0), :] / hs - np.dot(b[index(0), :], acc) + acc[index(0), :] / 2
        C1_prime = C1 - acc[index(0), :] / 2

        # calculate C2 by Eq. 85
        C2 = R[index(0), :] / hs ** 2 - np.dot(a[index(0), :], acc) + C1

        # calculate sn by Eq. 75 , calculate Sn by Eq. 86
        sn = np.zeros(shape)
        sn[index(0), :] = C1_prime
        Sn = np.zeros(shape)
        Sn[index(0), :] = C2 - C1

        for n in range(1, int(order / 2 + 1)):
            sn[index(n)] = sn[index(n - 1)] + 1 / 2 * (acc[index(n - 1)] + acc[index(n)])

            Sn[index(n)] = Sn[index(n - 1)] + sn[index(n - 1)] + 1 / 2 * acc[index(n - 1)]

        for n in range(-1, -int(order / 2 + 1), -1):
            sn[index(n)] = sn[index(n + 1)] - 1 / 2 * (acc[index(n + 1)] + acc[index(n)])

            Sn[index(n)] = Sn[index(n + 1)] - sn[index(n + 1)] + 1 / 2 * acc[index(n + 1)]

        V_update = np.zeros(shape)
        R_update = np.zeros(shape)

        # update the velocity and pos by Eq. 74 and Eq. 87
        for n in range(-int(order / 2), int(order / 2) + 1):
            R_update[index(n)] = hs ** 2 * (Sn[index(n)] + np.dot(a[index(n), :], acc))
            V_update[index(n)] = hs * (sn[index(n)] + np.dot(b[index(n), :], acc))

            # fix the central point
            R_update[index(0)] = R[index(0), :]
            V_update[index(0)] = V[index(0), :]

        acc_update = self.__evaluation(X, R_update, V_update)
        # print(X, R_update, V_update, acc_update, sn, Sn, self.__converge_test1(R, R_update, RelTol))
        return X, R_update, V_update, acc_update, sn, Sn, self.__converge_test1(R, R_update, RelTol)

    def __predictor(self, X: np.ndarray, R: np.ndarray, V: np.ndarray,
                    acc: np.ndarray, sn: np.ndarray, Sn: np.ndarray, hs):
        # PE

        a = self.__GJ_Coeff
        b = self.__Adam_Coef

        Sn_p = Sn[-1] + sn[-1] + 1 / 2 * acc[-1]

        R_p = hs ** 2 * (Sn_p + np.dot(a[-1, :], acc))

        V_p = hs * (sn[-1] + 1 / 2 * acc[-1] + np.dot(b[-1, :], acc))

        X_p = X[-1] + hs

        # evaluation
        acc_p = self.__func(X_p, R_p, V_p)
        # increment
        X = np.delete(X, 0, 0)
        R = np.delete(R, 0, 0)
        V = np.delete(V, 0, 0)
        Sn = np.delete(Sn, 0, 0)
        sn = np.delete(sn, 0, 0)
        acc = np.delete(acc, 0, 0)
        X = np.append(X, X_p)
        R = np.vstack((R, R_p))
        V = np.vstack((V, V_p))
        Sn = np.vstack((Sn, Sn_p))
        acc = np.vstack((acc, acc_p))
        sn = np.vstack((sn, np.zeros(np.shape(Sn_p))))
        return X, R, V, acc, Sn, sn

    def __corrector(self, X: np.ndarray, R: np.ndarray, V: np.ndarray,
                    acc: np.ndarray, sn: np.ndarray, Sn: np.ndarray, hs, rltot):

        # calculate sn
        a = self.__GJ_Coeff
        b = self.__Adam_Coef

        sn[-1] = sn[-2] + 1 / 2 * (acc[-1] + acc[-2])

        R_c = hs ** 2 * (Sn[-1] + np.dot(a[-2, :], acc))
        V_c = hs * (sn[-1] + np.dot(b[-2, :], acc))

        # evaluation
        acc_c = self.__func(X[-1], R_c, V_c)
        # update acc
        acc[-1] = acc_c

        # check convergence
        isConvergence = self.__converge_test3(R[-1], R_c, V[-1], V_c, rltot)

        # update R, V
        R[-1] = R_c
        V[-1] = V_c

        return R, V, acc, isConvergence

    def __evaluation(self, X, R, V):
        """
        Step 2 of Start-up
        Evaluate nine accelerations from initial positions and velocities, and those of epoch labelled
        (-4,-3,-2,-1,0,1,2,3,4)
        :param X: Time Epoch
        :param R: Pos
        :param V: Velocity
        :return:
        """
        acc = []

        for i in range(len(X)):
            temp1 = self.__func(X[i], R[i], V[i])
            acc.append(temp1)

        return np.array(acc)

    def __get_index(self, index_of_GJ):
        """
        :param index_of_GJ: index asked by GJ like (-4, -3, -2, -1, 0, 1, 2, 3, 4)
        As the label of GJ asks for a sequence like (-4, -3, -2, -1, 0, 1, 2, 3, 4) that is not consistent with the
         original index of list or array of PYTHON, we provide this function to bridge these two indexing.
        :return: index_of_python (0, 1, 2, 3, 4, 5, 6, 7, 8)
        """
        return int(self.__order / 2 + index_of_GJ)

    def __converge_test1(self, r0, r1, RelTol):
        """
        test convergence for two-dimension array
        :param r0:
        :param r1:
        :param RelTol:
        :return:
        """

        # shape = np.shape(r0)
        # v = r0 - r1
        # xx = []
        # for i in range(shape[1]):
        #     xx.append(self.__RMS(v[:, i]))
        #
        # err = self.__RMS(np.array(xx).flatten())

        err = np.max(np.abs(r0 - r1), 1)

        return np.max(err) < RelTol

    def __converge_test2(self, r0, r1, RelTol):
        """
        test convergence for one-dimension array
        :param r0:
        :param r1:
        :param RelTol:
        :return:
        """
        err = self.__RMS(r0 - r1)

        # err = np.abs(r0 - r1)

        return err < RelTol

    def __converge_test3(self, r0, r1, v0, v1, RelTol):
        """
        test convergence for one-dimension array
        :param r0:
        :param r1:
        :param RelTol:
        :return:
        """

        err = [np.fabs(r0 - r1), np.fabs(v0 - v1)]

        return np.max(np.max(err, 0)) < RelTol

    def __RMS(self, array):
        """
        Compute RMS of one-dimension array
        :param array:
        :return:
        """
        N = np.shape(array)[0]
        narray2 = array * array
        sum2 = narray2.sum()
        return np.sqrt(sum2 / N)

    @DeprecationWarning
    def getAttri(self):
        return self.__iniX, self.__iniR, self.__iniV, self.__evaluation, self.__startup, \
               self.__rltot, self.__predictor, self.__iterNum, self.__corrector

