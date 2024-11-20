import numpy as np
import cmath


class KeplerOrbit:
    """
    Most efficient way to propagate any type of two body orbit using kepler's equations.
    Reference:
    https://ww2.mathworks.cn/matlabcentral/fileexchange/36542-gauss-jackson-eighth-order-ode-solver-fixed-step-size
    """

    def __init__(self):

        self.setRecord(isRecord=True)
        pass

    def propagate(self, Nstep, hs):
        """

        :param Nstep: Steps to be integrated
        :param hs: stepsize
        :return:
        """

        X0 = self.__iniX
        R0 = self.__iniR
        V0 = self.__iniV

        t = np.array(range(0, hs * Nstep + hs, hs))
        shape = (len(t), len(R0))
        R = np.zeros(shape)
        V = np.zeros(shape)

        R[:] = R0
        V[:] = V0

        r_final, v_final = self.__keplerUniversal(R, V, t, self.__GM)

        if self.__isRecord:
            return t, r_final, v_final
        else:
            return t[-1], r_final[-1], v_final[-1]

    def setRecord(self, isRecord):
        self.__isRecord = isRecord
        return self

    def setInitial(self, iniX, iniR: np.ndarray, iniV: np.ndarray):
        """
        Define the initial value
        :param iniX: initial time epochs of given time series
        :param iniR: r (position) of given time series.
        :param iniV: v (velocity) of given time series
        :return:
        """

        self.__iniX = iniX
        self.__iniR = iniR
        self.__iniV = iniV

        return self

    def setArg(self, GM, rltot):
        self.__GM = GM
        self._tol = rltot
        return self

    def __keplerUniversal(self, r0, v0, t, mu):

        v0Mag = np.sqrt(np.sum(v0 ** 2, 1))
        r0Mag = np.sqrt(np.sum(r0 ** 2, 1))

        alpha = -(v0Mag ** 2) / mu + 2 / r0Mag

        # Compute initial guess (X0) for Newton's Method
        X0 = np.zeros(len(t))
        X0[:] = np.nan

        # Check if there are any Eliptic/Circular orbits
        idx = alpha > 0.00001

        if np.any(idx):
            X0[idx] = np.sqrt(mu) * t[idx] * alpha[idx]

        # Check if there are any Parabolic orbits
        idx = np.abs(alpha) < 0.000001
        if any(idx):
            h = np.cross(r0[idx, :], v0[idx, :])
            hMag = np.sqrt(np.sum(h ** 2, 1))
            p = (hMag ** 2) / mu
            s = (np.pi / 2 - np.arctan(3. * np.sqrt(mu / (p ** 3)) * t[idx])) / 2
            w = np.arctan(np.tan(s) ** (1 / 3))
            X0[idx] = np.sqrt(p) * 2. * 1 / np.tan(2. * w)

        # Check if there are any Hyperbolic orbits
        idx = alpha < -0.000001
        if any(idx):
            a = 1. / alpha[idx]

            term1 = 1 - r0Mag[idx] * alpha[idx]
            term2 = np.sign(t[idx]) * np.array([cmath.sqrt(-mu * a[i]) for i in range(len(t[idx]))])
            term3 = r0[idx, 0] * v0[idx, 0] + r0[idx, 1] * v0[idx, 1] + r0[idx, 2] * v0[idx, 2]
            term4 = -2. * mu * alpha[idx] * t[idx]
            term5 = np.sign(t[idx]) * np.array([cmath.sqrt(- a[i]) for i in range(len(t[idx]))])

            X0 = np.zeros(len(t), dtype=complex)
            X0[:] = np.nan
            X0[idx] = term5 * np.log(term4 / (term3 + term2 * term1))

        '''
        %% Newton's Method to converge on solution
        % Declare Constants that do not need to be computed within the while loop
        '''
        err = np.zeros(len(t))
        err[:] = np.inf
        dr0v0Smu = (r0[:, 0] * v0[:, 0] + r0[:, 1] * v0[:, 1] + r0[:, 2] * v0[:, 2]) / np.sqrt(mu)

        Smut = np.sqrt(mu) * t

        while any(abs(err) > self._tol):
            X02 = X0 ** 2
            X03 = X02 * X0
            psi = X02 * alpha
            [c2, c3] = self.__c2c3(psi)
            X0tOmPsiC3 = X0 * (1 - psi * c3)
            X02tC2 = X02 * c2
            r = X02tC2 + dr0v0Smu * X0tOmPsiC3 + r0Mag * (1 - psi * c2)
            Xn = X0 + (Smut - X03 * c3 - dr0v0Smu * X02tC2 - r0Mag * X0tOmPsiC3) / r
            err = Xn - X0
            X0 = Xn

        f = 1 - (Xn ** 2) * c2 / r0Mag
        g = t - (Xn ** 3) * c3 / np.sqrt(mu)
        gdot = 1 - c2 * (Xn ** 2) / r
        fdot = Xn * (psi * c3 - 1) * np.sqrt(mu) / (r * r0Mag)

        # r = bsxfun( @ times, f, r0) + bsxfun( @ times, g, v0);
        # v = bsxfun( @ times, fdot, r0) + bsxfun( @ times, gdot, v0);

        rf = np.zeros(np.shape(r0))
        vf = np.zeros(np.shape(v0))

        rf[:, 0] = f * r0[:, 0] + g * v0[:, 0]
        rf[:, 1] = f * r0[:, 1] + g * v0[:, 1]
        rf[:, 2] = f * r0[:, 2] + g * v0[:, 2]

        vf[:, 0] = fdot * r0[:, 0] + gdot * v0[:, 0]
        vf[:, 1] = fdot * r0[:, 1] + gdot * v0[:, 1]
        vf[:, 2] = fdot * r0[:, 2] + gdot * v0[:, 2]

        return rf, vf

    def __c2c3(self, psi):

        c2 = np.zeros(len(psi))
        c2[:] = np.nan
        c3 = np.zeros(len(psi))
        c3[:] = np.nan

        idx = psi > 1e-6
        if any(idx):
            c2[idx] = (1 - np.cos(np.sqrt(psi[idx]))) / psi[idx]
            c3[idx] = (np.sqrt(psi[idx]) - np.sin(np.sqrt(psi[idx]))) / np.sqrt(psi[idx] ** 3)

        idx = psi < -1e-6
        if any(idx):
            c2[idx] = (1 - np.cosh(np.sqrt(-psi[idx]))) / psi[idx]
            c3[idx] = (np.sinh(np.sqrt(-psi[idx])) - np.sqrt(-psi[idx])) / np.sqrt(-psi[idx] ** 3)

        idx = abs(psi) <= 1e-6
        if any(idx):
            c2[idx] = 0.5
            c3[idx] = 1 / 6

        return c2, c3