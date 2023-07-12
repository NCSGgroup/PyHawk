import numpy as np
from src.ODE.Common.ODEcoeffGenerator import ODEcoeffGenerator
from src.Preference.Pre_ODE import ODEConfig


class MultiStepOde:
    def __init__(self):
        self._ODEConfig = None
        self._MultiConfig = None
        pass

    def setForce(self, func):
        """
        Force function defined outside
        :param func: force function
        :return:
        """
        self._func = func
        return self

    def configure(self, ODEConfig:ODEConfig):
        self._ODEConfig = ODEConfig
        self._MultiConfig = self._ODEConfig.MultiStep()
        self._MultiConfig.__dict__.update(self._ODEConfig.MultiStepConfig.copy())
        return self

    def setRecord(self):
        """
        If isRecord=True, the orbit track (a list of vectors at every epoch) rather than
        the final epoch will be output.
        :param isRecord: Boolean
        :return:
        """
        self._isRecord = self._MultiConfig.isRecord
        return self

    def setInitial(self, iniX: list, iniR: list, iniV: list):
        """
        :param iniX: initial time epochs of given time series
        :param iniR: r (position) of given time series.
        :param iniV: v (velocity) of given time series
        :return:
        """
        return self

    def _loadCoeff(self):
        pass

    def propagate(self, N: int, hs):
        """
        The propagator starts to work for N steps from the given initial epoch
        :param N: Steps the propagator will extend
        :param hs: step size
        :return: OrbitX, OrbitR, OrbitV
        """
        assert N > 0
        OrbitX, OrbitR, OrbitV = [], [], []
        return OrbitX, OrbitR, OrbitV

    def propagate_one_step(self, hs, isFirstStep: True, dataIn):
        """
        One step forward once the function is called
        :param hs:
        :param isFirstStep: Mid-correction would be done if this is the first step of the whole integration
        :return:
        """
        pass


class ABM8(MultiStepOde):
    def __init__(self):
        super(ABM8, self).__init__()

    def _loadCoeff(self):
        """
               # This is the ordinate form (order by 8,7,6,5,4,3,2,1), see Table 4.5 (bash) and Table 4.7 (moul) ar ref 2
               The ordinate form can be also derived by Eq. (4.67) from coefficients in backward difference form.
               :return:
               """
        bash = [434241.0, -1152169.0, 2183877.0, -2664477.0, 2102243.0, -1041723.0, 295767.0, -36799.0]
        moul = [36799.0, 139849.0, -121797.0, 123133.0, -88547.0, 41499.0, -11351.0, 1375.0]
        self.__divisor = 1.0 / 120960.0
        self.__bash = np.array(bash)
        self.__moul = np.array(moul)

        pass

    def setInitial(self, iniX: list, iniR: list, iniV: list):
        """
        :param iniX: initial time epochs of given time series
        :param iniR: r (position) of given time series.
        :param iniV: v (velocity) of given time series
        :return:
        """
        assert len(iniX) == 8 and len(iniR) == 8 and len(iniV) == 8
        assert isinstance(iniR[-1], np.ndarray) and isinstance(iniV[-1], np.ndarray)

        self.__iniX = iniX
        self.__iniR = iniR
        self.__iniV = iniV
        return self

    def propagate(self, N: int, hs):
        """
        The propagator starts to work for N steps from the given initial epoch
        :param N: Steps the propagator will extend
        :param hs: step size
        :return: OrbitX, OrbitR, OrbitV
        """
        assert N > 0
        OrbitX, OrbitR, OrbitV = [], [], []

        X, R, V = self.__iniX, self.__iniR, self.__iniV
        Rdot, Vdot = self.__force_Evaluation(X, R, V)
        Xc, Rc, Vc = [], [], []
        for i in range(N):
            Xp, Rp, Vp = self.__predictor(X, R, V, hs, Rdot, Vdot)

            # the corrector could be repeated for times to achieve the convergence, but tests find the one-time
            # corrector is precise enough while preserving computation efficiency.
            Xc, Rc, Vc, Rdot, Vdot = self.__corrector(X, R, V, hs, Rdot, Vdot, Xp, Rp, Vp)

            # update the state vector
            X.pop(0)
            X.append(Xc)
            R.pop(0)
            R.append(Rc)
            V.pop(0)
            V.append(Vc)

            # update the track
            OrbitX.append(Xc)
            OrbitR.append(Rc)
            OrbitV.append(Vc)

        if self._isRecord:
            return OrbitX, OrbitR, OrbitV
        else:
            return Xc, Rc, Vc

    def __predictor(self, X, R, V, hs, Rdot, Vdot):
        """
        Calculate the predictor using the Adams-Bashforth formula
        :param X: current time epoch of given time series
        :param R: current pos of given time series
        :param V: current velocity of given time series
        :param hs: stepsize
        :return: Rp, Vp: predicted value at the next epoch
        """

        Rp = R[-1] + hs * self.__divisor * np.dot(np.flip(self.__bash), Rdot)
        Vp = V[-1] + hs * self.__divisor * np.dot(np.flip(self.__bash), Vdot)
        Xp = X[-1] + hs

        return Xp, Rp, Vp

    def __corrector(self, X, R, V, hs, Rdot, Vdot, Xp, Rp, Vp):
        """
        Calculate the corrector using the Adams-Moulton formula
        Note:
        In principle the third and fourth step would have to be repeated until convergence is achieved to
        ﬁnd the exact solution of the Adams–Moulton formula, but since each such iteration costs another
        function evaluation, this would not be worth the effort. A single corrector step is enough to assure
        that the order of the combined Adams–Bashforth–Moulton method is equal to that of the implicit method,
        even though the local truncation error is slightly larger (cf. Grigorieff 1977).
        :param X:
        :param R:
        :param V:
        :param hs:
        :param Rdot:
        :param Vdot:
        :param Rp: predicted value got from predictor
        :return:
        """

        Rdot_new, Vdot_new = self._func(Xp, Rp, Vp)
        Rdot = np.delete(np.vstack((Rdot, Rdot_new)), 0, axis=0)
        Vdot = np.delete(np.vstack((Vdot, Vdot_new)), 0, axis=0)

        Rc = R[-1] + hs * self.__divisor * np.dot(np.flip(self.__moul), Rdot)
        Vc = V[-1] + hs * self.__divisor * np.dot(np.flip(self.__moul), Vdot)

        # update the force matrix
        Rdot_new, Vdot_new = self._func(Xp, Rc, Vc)
        Rdot[-1, :] = Rdot_new
        Vdot[-1, :] = Vdot_new

        return Xp, Rc, Vc, Rdot, Vdot

    def __force_Evaluation(self, X, R, V):
        Rdot = []
        Vdot = []

        for i in range(len(X)):
            temp1, temp2 = self._func(X[i], R[i], V[i])
            Rdot.append(temp1)
            Vdot.append(temp2)

        return np.array(Rdot), np.array(Vdot)


class GaussJackson(MultiStepOde):
    def __init__(self):
        super(GaussJackson, self).__init__()
        self.order = None

    def setOrder(self, order: int):
        self.order = order
        assert np.mod(self.order, 2) == 0 and 4 <= self.order <= 18
        self._loadCoeff()
        return self

    def _loadCoeff(self):
        '''
        Generate or load the coefficients of propagator at given order
        :param order:
        :return: Adams coefficients and GJ coefficients
        '''
        dim = (self.order + 2, self.order + 1)
        ODE = ODEcoeffGenerator(self.order)
        Adam_Coef = np.array(ODE.SummedAdam_ordinate).astype(np.float64)
        GJ_Coeff = np.array(ODE.GaussJackson_ordinate).astype(np.float64)

        assert Adam_Coef.shape == dim and GJ_Coeff.shape == dim
        self.__Adam_Coef, self.__GJ_Coeff = Adam_Coef, GJ_Coeff

    def setIterArgs(self, iterNum: int, rltot):
        """
        :param rltot: tolerance of stopping an iteration.
        :param iterNum: define a maximal number of iteration to lower computation burden.
        :return:
        """
        self.__iterNum = iterNum
        self.__rltot = rltot
        return self

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
        assert len(iniX) == len(iniR) == len(iniV) == (self.order + 1)

        return self

    def propagate(self, N: int, hs):
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

        # if len(np.shape(R)) == 3:
        #     self.__mm= self.__startup1

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

            if self._isRecord:
                OrbitX.append(X[-1])
                OrbitR.append(R[-1])
                OrbitV.append(V[-1])

        if self._isRecord:
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
        order = self.order
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
        acc_p = self._func(X_p, R_p, V_p)
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
        acc_c = self._func(X[-1], R_c, V_c)
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
            temp1 = self._func(X[i], R[i], V[i])
            acc.append(temp1)

        return np.array(acc)

    def __get_index(self, index_of_GJ):
        """
        :param index_of_GJ: index asked by GJ like (-4, -3, -2, -1, 0, 1, 2, 3, 4)
        As the label of GJ asks for a sequence like (-4, -3, -2, -1, 0, 1, 2, 3, 4) that is not consistent with the
         original index of list or array of PYTHON, we provide this function to bridge these two indexing.
        :return: index_of_python (0, 1, 2, 3, 4, 5, 6, 7, 8)
        """
        return int(self.order / 2 + index_of_GJ)

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


class MatrixGaussJackson(MultiStepOde):
    def __init__(self):
        super(MatrixGaussJackson, self).__init__()
        self.order = None

    def setOrder(self, order: int):
        self.order = order
        assert np.mod(self.order, 2) == 0 and 4 <= self.order <= 18
        self._loadCoeff()
        return self

    def _loadCoeff(self):
        '''
        Generate or load the coefficients of propagator at given order
        :param order:
        :return: Adams coefficients and GJ coefficients
        '''
        dim = (self.order + 2, self.order + 1)
        ODE = ODEcoeffGenerator(self.order)
        Adam_Coef = np.array(ODE.SummedAdam_ordinate).astype(np.float64)
        GJ_Coeff = np.array(ODE.GaussJackson_ordinate).astype(np.float64)

        assert Adam_Coef.shape == dim and GJ_Coeff.shape == dim
        self.__Adam_Coef, self.__GJ_Coeff = Adam_Coef, GJ_Coeff

    def setIterArgs(self, iterNum: int, rltot):
        """
        :param rltot: tolerance of stopping an iteration.
        :param iterNum: define a maximal number of iteration to lower computation burden.
        :return:
        """
        self.__iterNum = iterNum
        self.__rltot = rltot
        return self

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

        assert len(iniX) == len(iniR) == len(iniV) == (self.order + 1)

        return self

    def propagate(self, N: int, hs):
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

        # if len(np.shape(R)) == 3:
        #     self.__mm= self.__startup1

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

            if self._isRecord:
                OrbitX.append(X[-1])
                OrbitR.append(R[-1])
                OrbitV.append(V[-1])

        if self._isRecord:
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
        order = self.order
        b = self.__Adam_Coef
        a = self.__GJ_Coeff

        dim = np.shape(X)[0]
        npar = np.shape(R)[1]
        mpar = np.shape(R)[2]
        shape = (dim, npar, mpar)

        assert np.shape(R) == np.shape(V) == np.shape(acc) == shape
        assert dim == order + 1

        # calculate C1 by Eq. 73 in ref 1
        # print(V[index(0), :])
        # print(V[index(0)])

        C1 = V[index(0), :] / hs - np.einsum('m,mln->ln', b[index(0), :], acc) + acc[index(0), :] / 2
        C1_prime = C1 - acc[index(0), :] / 2
        # calculate C2 by Eq. 85
        C2 = R[index(0), :] / hs ** 2 - np.einsum('m,mln->ln', a[index(0), :], acc) + C1

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
            R_update[index(n)] = hs ** 2 * (Sn[index(n)] + np.einsum('m,mln->ln', a[index(n), :], acc))
            V_update[index(n)] = hs * (sn[index(n)] + np.einsum('m,mln->ln', b[index(n), :], acc))

            # fix the central point
            R_update[index(0)] = R[index(0), :]
            V_update[index(0)] = V[index(0), :]

        acc_update = self.__evaluation(X, R_update, V_update)
        # acc_update = self.__evaluation(X, R_update, V_update)
        # print(X, R_update, V_update, acc_update, sn, Sn, self.__converge_test1(R, R_update, RelTol))
        return X, R_update, V_update, acc_update, sn, Sn, self.__converge_test1(R, R_update, RelTol)

    def __predictor(self, X: np.ndarray, R: np.ndarray, V: np.ndarray,
                    acc: np.ndarray, sn: np.ndarray, Sn: np.ndarray, hs):
        # PE

        a = self.__GJ_Coeff
        b = self.__Adam_Coef

        Sn_p = Sn[-1] + sn[-1] + 1 / 2 * acc[-1]

        R_p = hs ** 2 * (Sn_p + np.einsum('m,mln->ln', a[-1, :], acc))

        V_p = hs * (sn[-1] + 1 / 2 * acc[-1] + np.einsum('m,mln->ln', b[-1, :], acc))

        X_p = X[-1] + hs

        # evaluation
        acc_p = self._func(X_p, R_p, V_p)
        # acc_p = self._func(X_p, R_p, V_p)
        # increment
        X = np.delete(X, 0, 0)
        R = np.delete(R, 0, 0)
        V = np.delete(V, 0, 0)
        Sn = np.delete(Sn, 0, 0)
        sn = np.delete(sn, 0, 0)
        acc = np.delete(acc, 0, 0)
        X = np.append(X, X_p)
        R = np.vstack((R, R_p[None, :, :]))
        V = np.vstack((V, V_p[None, :, :]))
        Sn = np.vstack((Sn, Sn_p[None, :, :]))
        acc = np.vstack((acc, acc_p[None, :, :]))
        sn = np.vstack((sn, np.zeros(np.shape(Sn_p))[None, :, :]))
        return X, R, V, acc, Sn, sn

    def __corrector(self, X: np.ndarray, R: np.ndarray, V: np.ndarray,
                    acc: np.ndarray, sn: np.ndarray, Sn: np.ndarray, hs, rltot):

        # calculate sn
        a = self.__GJ_Coeff
        b = self.__Adam_Coef

        sn[-1] = sn[-2] + 1 / 2 * (acc[-1] + acc[-2])
        np.einsum('m,mln->ln', a[-1, :], acc)
        R_c = hs ** 2 * (Sn[-1] + np.einsum('m,mln->ln', a[-2, :], acc))
        V_c = hs * (sn[-1] + np.einsum('m,mln->ln', b[-2, :], acc))

        # evaluation
        acc_c = self._func(X[-1], R_c, V_c)
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
            temp1 = self._func(X[i], R[i], V[i])
            acc.append(temp1)

        return np.array(acc)

    def __get_index(self, index_of_GJ):
        """
        :param index_of_GJ: index asked by GJ like (-4, -3, -2, -1, 0, 1, 2, 3, 4)
        As the label of GJ asks for a sequence like (-4, -3, -2, -1, 0, 1, 2, 3, 4) that is not consistent with the
         original index of list or array of PYTHON, we provide this function to bridge these two indexing.
        :return: index_of_python (0, 1, 2, 3, 4, 5, 6, 7, 8)
        """
        return int(self.order / 2 + index_of_GJ)

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