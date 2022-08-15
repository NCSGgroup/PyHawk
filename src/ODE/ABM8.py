import numpy as np


class ABM8:
    """
     An orbit propagator with multiple-step integration method: Adams-Bashforth-Moulton 8th order
     Reference:
     1. https://ww2.mathworks.cn/matlabcentral/fileexchange/55433-adams-bashforth-moulton
     2 Book: Satellite Orbits: Models, Methods and Applications Chapter 4
    """

    def __init__(self):
        self.setRecord()
        self.__loadCoeff()
        pass

    def setForce(self, func):
        self.__func = func
        return self

    def setRecord(self, isRecord=True):
        """
        If isRecord=True, the orbit track (a list of vectors at every epoch) rather than
        the final epoch will be output.
        :param isRecord: Boolean
        :return:
        """
        self.__isRecord = isRecord
        return self

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

    def __loadCoeff(self):
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

        if self.__isRecord:
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

        Rdot_new, Vdot_new = self.__func(Xp, Rp, Vp)
        Rdot = np.delete(np.vstack((Rdot, Rdot_new)), 0, axis=0)
        Vdot = np.delete(np.vstack((Vdot, Vdot_new)), 0, axis=0)

        Rc = R[-1] + hs * self.__divisor * np.dot(np.flip(self.__moul), Rdot)
        Vc = V[-1] + hs * self.__divisor * np.dot(np.flip(self.__moul), Vdot)

        # update the force matrix
        Rdot_new, Vdot_new = self.__func(Xp, Rc, Vc)
        Rdot[-1, :] = Rdot_new
        Vdot[-1, :] = Vdot_new

        return Xp, Rc, Vc, Rdot, Vdot

    def __force_Evaluation(self, X, R, V):
        Rdot = []
        Vdot = []

        for i in range(len(X)):
            temp1, temp2 = self.__func(X[i], R[i], V[i])
            Rdot.append(temp1)
            Vdot.append(temp2)

        return np.array(Rdot), np.array(Vdot)