import sys
sys.path.append('../')
from src.ForceModel.BaseSecondDerivative import BaseSecondDerivative
import numpy as np
import warnings
import platform
from ctypes import cdll
from ctypes import c_int, c_double
from numpy.ctypeslib import ndpointer
import pathlib as _pathlib
import glob as _glob
from src.Preference.Pre_ForceModel import ForceModelConfig


class AbsGravHar(BaseSecondDerivative):
    def __init__(self):
        super(AbsGravHar, self).__init__()
        self._GM = None
        self._Radius = None
        self._Nmax = None
        self._pos = None
        self._Cpot = None
        self._Spot = None
        self._FMConfig = None
        self._GravHarConfig = None

    def configure(self, FMConfig: ForceModelConfig):
        return self

    def setPar(self, Cpot: np.ndarray, Spot: np.ndarray, Pos: np.ndarray):
        return self

    def get_potential(self):
        pass


class PyGravHar(AbsGravHar):

    def __init__(self):
        '''
        :param GM: Geopotential constant of the Earth in unit of [m^3/s^2]
        :param Radius: The radius of the Earth [m]
        '''
        super(PyGravHar, self).__init__()
        # Above value are not quite precise, suggesting 'setConstant'
        # self.setConstant()
        pass

    def setPar(self, Cpot: np.ndarray, Spot: np.ndarray, Pos: np.ndarray):
        """
        Spherical harmonic coefficients in one-dimension (in terms of potential or geoid-height)
        :param Cpot: CPOT((N_max+1)*(N_max+2)/2),SPOT((N_max+1)*(N_max+2)/2)
        :param Spot:
        :param Pos: position in ITRS frame, a three dimension array
        :return:
        """
        assert len(Pos) == 3
        assert len(Cpot) == len(Spot)
        L = len(Cpot)
        N = int(np.round(np.sqrt(2 * L))) - 1
        assert (N + 1) * (N + 2) / 2 == L

        if self._Nmax != N:
            self._Nmax = N
            self.__preprocess(N)

        self.__calVW(Pos)
        self._Cpot = Cpot
        self._Spot = Spot
        self._pos = Pos

        return self

    def __calVW(self, pos):
        """
        To calculate and store the V and W for later use
        this function should be called once the Pos or CS has been changed.
        """

        (f1meq0, f2meq0, f1meq1, f2meq1, f3meq1, grdtcn1, grdtcn2, grdtcn3, grdtcn4,
         grdtcn5, grdtcn6, grdtcn7, grdtcn8, grdtcn9, grdtcn10, grdtcn11, grdtcn12, vwmeq1, vwn1, vwn2,
         f1mgt1, f2mgt1, f3mgt1, grdtcnm1, grdtcnm2, grdtcnm3, grdtcnm4, grdtcnm5, grdtcnm6, vwnm1,
         vwnm2) = self.__coeff

        xyz = pos

        numpar = int((self._Nmax + 1) * (self._Nmax + 2) / 2)

        R_SQR = xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2

        rho = self._Radius ** 2 / R_SQR

        # Normalized coordinates
        [x0, y0, z0] = self._Radius * xyz / R_SQR
        '''
        ! Evaluate harmonic functions 
        ! V_nm_par =N_nm/N_m-1_m-1*(2m-1)*(R_ref/r)^(n+1) * P_nm(sin(phi)) * cos(m*lambda)
        ! and 
        ! W_nm = (R_ref/r)^(n+1) * P_nm(sin(phi)) * sin(m*lambda)
        ! up to degree and order n_max+1
        '''

        # Calculate zonal terms V(n,0); set W(n,0)=0.0

        V = np.zeros(shape=(self._Nmax + 3, self._Nmax + 3), dtype=np.float64)
        W = np.zeros(shape=(self._Nmax + 3, self._Nmax + 3), dtype=np.float64)

        V[0, 0] = self._Radius / np.sqrt(R_SQR)
        V[1, 0] = np.sqrt(3) * z0 * V[0, 0]

        for n in range(2, self._Nmax + 3):
            V[n, 0] = vwn1[n] * z0 * V[n - 1, 0] - vwn2[n] * rho * V[n - 2, 0]

        W[:, 0] = 0.

        # Calculate tesseral and sectorial terms

        for m in range(1, self._Nmax + 3):
            # Calculate V(m,m) .. V(n_max+1,m)
            if m == 1:
                V[1, 1] = np.sqrt(3.) * x0 * V[0, 0]
                W[1, 1] = np.sqrt(3.) * y0 * V[0, 0]
            else:
                V[m, m] = vwmeq1[m] * (x0 * V[m - 1, m - 1] - y0 * W[m - 1, m - 1])
                W[m, m] = vwmeq1[m] * (x0 * W[m - 1, m - 1] + y0 * V[m - 1, m - 1])

            if m <= self._Nmax + 1:
                V[m + 1, m] = np.sqrt((2 * m + 3)) * z0 * V[m, m]
                W[m + 1, m] = np.sqrt((2 * m + 3)) * z0 * W[m, m]

            for n in range(m + 2, self._Nmax + 3):
                index = int(n * (n + 1) / 2 + m)
                V[n, m] = vwnm1[index] * z0 * V[n - 1, m] - vwnm2[index] * rho * V[n - 2, m]
                W[n, m] = vwnm1[index] * z0 * W[n - 1, m] - vwnm2[index] * rho * W[n - 2, m]

        self.__V, self.__W = V, W

        return True

    def __preprocess(self, Nmax):
        '''
        Compute the relevant coefficients and store them for later use.
        Vectorized
        :return:
        '''
        # TODO: recover warnings
        warnings.simplefilter('ignore')
        n = np.arange(Nmax + 3).astype(float)

        f1meq0 = np.sqrt((2 * n + 1) * (n + 2) * (n + 1) / (4 * n + 6))
        f2meq0 = (n + 1) * np.sqrt((2 * n + 1) / (2 * n + 3))
        f1meq1 = np.sqrt((2 * n + 1) * (n + 2) * (n + 3) / (2 * n + 3))
        f2meq1 = np.sqrt((2 * n + 1) * (n + 1) * n * 2 / (2 * n + 3))
        f3meq1 = np.sqrt((2 * n + 1) * (n + 2) * n / (2 * n + 3))

        # for tensor coefficient
        grdtcn1 = np.sqrt((2 * n + 1) * (n + 1) * (n + 2) * (n + 3) * (n + 4) / (4 * n + 10))
        grdtcn2 = (n + 1) * np.sqrt((2 * n + 1) * (n + 2) * (n + 3) / (4 * n + 10))
        grdtcn3 = (n + 1) * (n + 2) * np.sqrt((2 * n + 1) / (2 * n + 5))
        grdtcn4 = np.sqrt((2 * n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) / (2 * n + 5))
        grdtcn5 = np.sqrt((2 * n + 1) * n * (n + 1) * (n + 2) * (n + 3) / (2 * n + 5))
        grdtcn6 = np.sqrt(n * (n + 2) * (n + 3) * (n + 4) * (2 * n + 1) / (2 * n + 5))
        grdtcn7 = (n + 2) * np.sqrt(2 * (2 * n + 1) * (n + 1) * n / (2 * n + 5))
        grdtcn8 = np.sqrt((n + 3) * (n + 4) * (n + 5) * (n + 6) * (2 * n + 1) / (2 * n + 5))
        grdtcn9 = 2 * np.sqrt((n - 1) * n * (n + 3) * (n + 4) * (2 * n + 1) / (2 * n + 5))
        grdtcn10 = np.sqrt((n - 1) * n * (n + 1) * (n + 2) * (4 * n + 2) / (2 * n + 5))
        grdtcn11 = np.sqrt((n - 1) * (n + 3) * (n + 4) * (n + 5) * (2 * n + 1) / (2 * n + 5))  # n, m
        grdtcn12 = np.sqrt((n + 3) * (n - 1) * n * (n + 1) * (2 * n + 1) / (2 * n + 5))

        # coefficient for V, W
        vwmeq1 = np.sqrt((2 * n + 1) / (2 * n))
        vwn1 = np.sqrt((2 * n + 1) / (n ** 2)) * np.sqrt((2 * n - 1))  # avwmeq0(n)
        vwn2 = np.sqrt((2 * n + 1) / (n ** 2)) * (n - 1) / np.sqrt((2 * n - 3))  # bvwmeq0(n)

        f1mgt1 = []
        f2mgt1 = []
        f3mgt1 = []
        grdtcnm1 = []
        grdtcnm2 = []
        grdtcnm3 = []
        grdtcnm4 = []
        grdtcnm5 = []
        grdtcnm6 = []
        vwnm1 = []
        vwnm2 = []

        for n in range(Nmax + 3):
            m = np.arange(n + 1).astype(np.float64)
            # index = n * (n + 1) / 2 + m + 1
            # force coefficient
            f1mgt1_a = np.sqrt((2 * n + 1) * (n + m + 2) * (n + m + 1) / (2 * n + 3))
            f2mgt1_a = np.sqrt((2 * n + 1) * (n - m + 2) * (n - m + 1) / (2 * n + 3))
            f3mgt1_a = np.sqrt((2 * n + 1) * (n + m + 1) * (n - m + 1) / (2 * n + 3))

            # tensor coefficient
            grdtcnm1_a = np.sqrt((n + m + 1) * (n + m + 2) * (n + m + 3) * (n + m + 4) * (2 * n + 1) / (2 * n + 5))
            grdtcnm2_a = np.sqrt((n - m + 1) * (n + m + 1) * (n + m + 2) * (n + m + 3) * (2 * n + 1) / (2 * n + 5))
            grdtcnm3_a = 2. * np.sqrt((n - m + 1) * (n - m + 2) * (n + m + 1) * (n + m + 2) * (2 * n + 1) / (2 * n + 5))
            grdtcnm4_a = np.sqrt((n - m + 4) * (n - m + 1) * (n - m + 2) * (n - m + 3) * (2 * n + 1) / (2 * n + 5))
            grdtcnm5_a = np.sqrt((n + m + 1) * (n - m + 1) * (n - m + 2) * (n - m + 3) * (2 * n + 1) / (2 * n + 5))
            grdtcnm6_a = np.sqrt((2 * n + 1) * (n + m + 1) * (n + m + 2) * (n - m + 1) * (n - m + 2) / (2 * n + 5))

            # coefficient for V, M
            vwnm1_a = np.sqrt((2 * n + 1) / (n ** 2 - m ** 2)) * np.sqrt((2 * n - 1))
            vwnm2_a = np.sqrt((2 * n + 1) / (n ** 2 - m ** 2)) * np.sqrt(((n - m - 1) * (n + m - 1)) / (2 * n - 3))

            f1mgt1 = np.append(f1mgt1, f1mgt1_a)
            f2mgt1 = np.append(f2mgt1, f2mgt1_a)
            f3mgt1 = np.append(f3mgt1, f3mgt1_a)
            grdtcnm1 = np.append(grdtcnm1, grdtcnm1_a)
            grdtcnm2 = np.append(grdtcnm2, grdtcnm2_a)
            grdtcnm3 = np.append(grdtcnm3, grdtcnm3_a)
            grdtcnm4 = np.append(grdtcnm4, grdtcnm4_a)
            grdtcnm5 = np.append(grdtcnm5, grdtcnm5_a)
            grdtcnm6 = np.append(grdtcnm6, grdtcnm6_a)
            vwnm1 = np.append(vwnm1, vwnm1_a)
            vwnm2 = np.append(vwnm2, vwnm2_a)

        self.__coeff = (f1meq0, f2meq0, f1meq1, f2meq1, f3meq1, grdtcn1, grdtcn2, grdtcn3, grdtcn4,
                        grdtcn5, grdtcn6, grdtcn7, grdtcn8, grdtcn9, grdtcn10, grdtcn11, grdtcn12, vwmeq1, vwn1, vwn2,
                        f1mgt1, f2mgt1, f3mgt1, grdtcnm1, grdtcnm2, grdtcnm3, grdtcnm4, grdtcnm5, grdtcnm6, vwnm1,
                        vwnm2)
        pass

    def get_potential(self):
        """
        :return: Geopotential of the given position
        """

        V, W = self.__V, self.__W
        U_pot = 0.

        for m in range(self._Nmax + 1):
            for n in range(m, self._Nmax + 1):
                Index = int(n * (n + 1) / 2 + m)
                U_pot += self._Cpot[Index] * V[n, m] + self._Spot[Index] * W[n, m]

        U_pot = self._GM / self._Radius * U_pot

        return U_pot

    def getAcceleration(self):
        """
        [m/s^2] in ITRS frame
        Notice, this is only the acc due to geopential rather than the true gravity acceleration
        :return: three-dimensional accelerations a_x, a_y, a_z
        """

        acc = np.zeros(3, dtype=np.float64)

        V, W = self.__V, self.__W
        (f1meq0, f2meq0, f1meq1, f2meq1, f3meq1, grdtcn1, grdtcn2, grdtcn3, grdtcn4,
         grdtcn5, grdtcn6, grdtcn7, grdtcn8, grdtcn9, grdtcn10, grdtcn11, grdtcn12, vwmeq1, vwn1, vwn2,
         f1mgt1, f2mgt1, f3mgt1, grdtcnm1, grdtcnm2, grdtcnm3, grdtcnm4, grdtcnm5, grdtcnm6, vwnm1,
         vwnm2) = self.__coeff

        Nmax = self._Nmax
        for m in range(Nmax + 1):
            for n in range(m, Nmax + 1):
                index = int(n * (n + 1) / 2 + m)
                C = self._Cpot[index]
                S = self._Spot[index]

                if m == 0:
                    acc[0] += - C * V[n + 1, 1] * f1meq0[n]
                    acc[1] += - C * W[n + 1, 1] * f1meq0[n]
                    acc[2] += - C * V[n + 1, 0] * f2meq0[n]
                elif m == 1:
                    acc[0] += 0.5 * (f1meq1[n] * (- C * V[n + 1, 2] - S * W[n + 1, 2]) + f2meq1[n] * C * V[n + 1, 0])
                    acc[1] += 0.5 * (f1meq1[n] * (- C * W[n + 1, 2] + S * V[n + 1, 2]) + f2meq1[n] * S * V[n + 1, 0])
                    acc[2] += f3meq1[n] * (- C * V[n + 1, 1] - S * W[n + 1, 1])
                else:
                    acc[0] += 0.5 * (f1mgt1[index] * (- C * V[n + 1, m + 1] - S * W[n + 1, m + 1]) +
                                     f2mgt1[index] * (+ C * V[n + 1, m - 1] + S * W[n + 1, m - 1]))
                    acc[1] += 0.5 * (f1mgt1[index] * (- C * W[n + 1, m + 1] + S * V[n + 1, m + 1]) +
                                     f2mgt1[index] * (- C * W[n + 1, m - 1] + S * V[n + 1, m - 1]))
                    acc[2] += f3mgt1[index] * (- C * V[n + 1, m] - S * W[n + 1, m])

        GM = self._GM
        R_ref = self._Radius
        acc = (GM / (R_ref * R_ref)) * acc

        return acc

    def getAccelerationGCRS(self):
        """
        [m/s^2] in ITRS frame
        Notice, this is only the acc due to geopential rather than the true gravity acceleration
        :return: three-dimensional accelerations a_x, a_y, a_z
        """

        acc = np.zeros(3, dtype=np.float64)

        V, W = self.__V, self.__W
        (f1meq0, f2meq0, f1meq1, f2meq1, f3meq1, grdtcn1, grdtcn2, grdtcn3, grdtcn4,
         grdtcn5, grdtcn6, grdtcn7, grdtcn8, grdtcn9, grdtcn10, grdtcn11, grdtcn12, vwmeq1, vwn1, vwn2,
         f1mgt1, f2mgt1, f3mgt1, grdtcnm1, grdtcnm2, grdtcnm3, grdtcnm4, grdtcnm5, grdtcnm6, vwnm1,
         vwnm2) = self.__coeff

        Nmax = self._Nmax
        for m in range(Nmax + 1):
            for n in range(m, Nmax + 1):
                index = int(n * (n + 1) / 2 + m)
                C = self._Cpot[index]
                S = self._Spot[index]

                if m == 0:
                    acc[0] += - C * V[n + 1, 1] * f1meq0[n]
                    acc[1] += - C * W[n + 1, 1] * f1meq0[n]
                    acc[2] += - C * V[n + 1, 0] * f2meq0[n]
                elif m == 1:
                    acc[0] += 0.5 * (f1meq1[n] * (- C * V[n + 1, 2] - S * W[n + 1, 2]) + f2meq1[n] * C * V[n + 1, 0])
                    acc[1] += 0.5 * (f1meq1[n] * (- C * W[n + 1, 2] + S * V[n + 1, 2]) + f2meq1[n] * S * V[n + 1, 0])
                    acc[2] += f3meq1[n] * (- C * V[n + 1, 1] - S * W[n + 1, 1])
                else:
                    acc[0] += 0.5 * (f1mgt1[index] * (- C * V[n + 1, m + 1] - S * W[n + 1, m + 1]) +
                                     f2mgt1[index] * (+ C * V[n + 1, m - 1] + S * W[n + 1, m - 1]))
                    acc[1] += 0.5 * (f1mgt1[index] * (- C * W[n + 1, m + 1] + S * V[n + 1, m + 1]) +
                                     f2mgt1[index] * (- C * W[n + 1, m - 1] + S * V[n + 1, m - 1]))
                    acc[2] += f3mgt1[index] * (- C * V[n + 1, m] - S * W[n + 1, m])

        GM = self._GM
        R_ref = self._Radius
        acc = (GM / (R_ref * R_ref)) * acc

        return acc

    def getDaDx(self):
        """
        in [ITRS] frame
        A SYMMETRIC matrix is returned as below
                ax/dx ay/dx az/dx         ax/dx ax/dy ax/dz
                ...   ay/dy az/dy  =            ay/dy ay/dz
                ...   ...   az/dz                     az/dz
        Compute the partial derivative of acceleration with respect to the position
        :return:da_x/dx da_y/dy da_z/dz = Vxx, Vxy, Vxz , Vyx, Vyy, Vyz , Vzx, Vzy, Vzz
        """
        V, W = self.__V, self.__W
        (f1meq0, f2meq0, f1meq1, f2meq1, f3meq1, grdtcn1, grdtcn2, grdtcn3, grdtcn4,
         grdtcn5, grdtcn6, grdtcn7, grdtcn8, grdtcn9, grdtcn10, grdtcn11, grdtcn12, vwmeq1, vwn1, vwn2,
         f1mgt1, f2mgt1, f3mgt1, grdtcnm1, grdtcnm2, grdtcnm3, grdtcnm4, grdtcnm5, grdtcnm6, vwnm1,
         vwnm2) = self.__coeff

        DU2XYZ = np.zeros((3, 3))

        for m in range(self._Nmax + 1):
            for n in range(m, self._Nmax + 1):

                index = int(n * (n + 1) / 2 + m)
                C = self._Cpot[index]
                S = self._Spot[index]

                if m == 0:
                    DU2XYZ[0, 0] += 0.5 * (grdtcn1[n] * C * V[n + 2, 2] - grdtcn3[n] * C * V[n + 2, 0])
                    DU2XYZ[0, 1] += 0.5 * grdtcn1[n] * C * W[n + 2, 2]
                    DU2XYZ[0, 2] += grdtcn2[n] * C * V[n + 2, 1]
                    DU2XYZ[1, 2] += grdtcn2[n] * C * W[n + 2, 1]
                    DU2XYZ[2, 2] += grdtcnm6[index] * C * V[n + 2, 0]
                elif m == 1:
                    DU2XYZ[0, 0] += 0.25 * (grdtcn4[n] * (C * V[n + 2, 3] + S * W[n + 2, 3]) -
                                            grdtcn5[n] * (3. * C * V[n + 2, 1] + S * W[n + 2, 1]))
                    DU2XYZ[0, 1] += 0.25 * (grdtcn4[n] * (C * W[n + 2, 3] - S * V[n + 2, 3]) +
                                            grdtcn5[n] * (- C * W[n + 2, 1] - S * V[n + 2, 1]))
                    DU2XYZ[0, 2] += 0.5 * (grdtcn6[n] * (C * V[n + 2, 2] + S * W[n + 2, 2]) -
                                           grdtcn7[n] * C * V[n + 2, 0])
                    DU2XYZ[1, 2] += 0.5 * (grdtcn6[n] * (C * W[n + 2, 2] - S * V[n + 2, 2]) -
                                           grdtcn7[n] * S * V[n + 2, 0])
                    DU2XYZ[2, 2] += grdtcnm6[index] * (C * V[n + 2, 1] + S * W[n + 2, 1])
                elif m == 2:
                    DU2XYZ[0, 0] += 0.25 * (grdtcn8[n] * (+ C * V[n + 2, 4] + S * W[n + 2, 4]) +
                                            grdtcn9[n] * (- C * V[n + 2, 2] - S * W[n + 2, 2]) +
                                            grdtcn10[n] * C * V[n + 2, 0])
                    DU2XYZ[0, 1] += 0.25 * (grdtcn8[n] * (+ C * W[n + 2, m + 2] - S * V[n + 2, m + 2]) +
                                            grdtcn10[n] * (- C * W[n + 2, m - 2] + S * V[n + 2, m - 2]))
                    DU2XYZ[0, 2] += 0.5 * (grdtcn11[n] * (+ C * V[n + 2, m + 1] + S * W[n + 2, m + 1]) +
                                           grdtcn12[n] * (- C * V[n + 2, m - 1] - S * W[n + 2, m - 1]))
                    DU2XYZ[1, 2] += 0.5 * (grdtcn11[n] * (+ C * W[n + 2, m + 1] - S * V[n + 2, m + 1]) +
                                           grdtcn12[n] * (+ C * W[n + 2, m - 1] - S * V[n + 2, m - 1]))
                    DU2XYZ[2, 2] += grdtcnm6[index] * (+ C * V[n + 2, m] + S * W[n + 2, m])
                else:
                    DU2XYZ[0, 0] += 0.25 * (grdtcnm1[index] * (+ C * V[n + 2, m + 2] + S * W[n + 2, m + 2]) +
                                            grdtcnm3[index] * (- C * V[n + 2, m] - S * W[n + 2, m]) +
                                            grdtcnm4[index] * (+ C * V[n + 2, m - 2] + S * W[n + 2, m - 2]))
                    DU2XYZ[0, 1] += 0.25 * (grdtcnm1[index] * (+ C * W[n + 2, m + 2] - S * V[n + 2, m + 2]) +
                                            grdtcnm4[index] * (- C * W[n + 2, m - 2] + S * V[n + 2, m - 2]))
                    DU2XYZ[0, 2] += 0.5 * (grdtcnm2[index] * (+ C * V[n + 2, m + 1] + S * W[n + 2, m + 1]) +
                                           grdtcnm5[index] * (- C * V[n + 2, m - 1] - S * W[n + 2, m - 1]))
                    DU2XYZ[1, 2] += 0.5 * (grdtcnm2[index] * (+ C * W[n + 2, m + 1] - S * V[n + 2, m + 1]) +
                                           grdtcnm5[index] * (+ C * W[n + 2, m - 1] - S * V[n + 2, m - 1]))
                    DU2XYZ[2, 2] += grdtcnm6[index] * (+ C * V[n + 2, m] + S * W[n + 2, m])

        GM = self._GM
        R_ref = self._Radius

        DU2XYZ[0, 0] = GM / R_ref ** 3 * DU2XYZ[0, 0]
        DU2XYZ[0, 1] = GM / R_ref ** 3 * DU2XYZ[0, 1]
        DU2XYZ[0, 2] = GM / R_ref ** 3 * DU2XYZ[0, 2]
        DU2XYZ[1, 2] = GM / R_ref ** 3 * DU2XYZ[1, 2]
        DU2XYZ[2, 2] = GM / R_ref ** 3 * DU2XYZ[2, 2]
        DU2XYZ[1, 0] = DU2XYZ[0, 1]
        DU2XYZ[2, 1] = DU2XYZ[1, 2]
        DU2XYZ[2, 0] = DU2XYZ[0, 2]
        DU2XYZ[1, 1] = - DU2XYZ[0, 0] - DU2XYZ[2, 2]

        return DU2XYZ

    def getDaDp(self, degree_max: int, degree_min):
        """
        in [ITRS] frame
        Compute the partial derivative of acceleration with respect to harmonic coefficients
        :param degree_max:max degree of the harmonic coefficients to be estimated.
        :param degree_min:minimal degree of the harmonic coefficients to be estimated. =1 by default
        :return: da/dc da/ds starting from n=0 !!notice this
        """

        assert 1 <= degree_min <= degree_max <= self._Nmax
        du2c = np.zeros((3, int((degree_max + 1) * (degree_max + 2) / 2)), dtype=np.float64)
        du2s = np.zeros((3, int((degree_max + 1) * (degree_max + 2) / 2)), dtype=np.float64)

        GM = self._GM
        R_ref = self._Radius
        factor = (GM / (R_ref * R_ref))
        V, W = self.__V, self.__W
        (f1meq0, f2meq0, f1meq1, f2meq1, f3meq1, grdtcn1, grdtcn2, grdtcn3, grdtcn4,
         grdtcn5, grdtcn6, grdtcn7, grdtcn8, grdtcn9, grdtcn10, grdtcn11, grdtcn12, vwmeq1, vwn1, vwn2,
         f1mgt1, f2mgt1, f3mgt1, grdtcnm1, grdtcnm2, grdtcnm3, grdtcnm4, grdtcnm5, grdtcnm6, vwnm1,
         vwnm2) = self.__coeff

        for m in range(degree_max + 1):
            for n in range(m, degree_max + 1):

                index = int(n * (n + 1) / 2 + m)

                if degree_min <= n:
                    if m == 0:
                        du2c[0, index] = -factor * V[n + 1, 1] * f1meq0[n]
                        du2c[1, index] = -factor * W[n + 1, 1] * f1meq0[n]
                        du2c[2, index] = -factor * V[n + 1, 0] * f2meq0[n]
                    elif m == 1:
                        du2c[0, index] = 0.5 * factor * (- V[n + 1, 2] * f1meq1[n] + V[n + 1, 0] * f2meq1[n])
                        du2s[0, index] = 0.5 * factor * (- W[n + 1, 2] * f1meq1[n])
                        du2c[1, index] = 0.5 * factor * (- W[n + 1, 2] * f1meq1[n])
                        du2s[1, index] = 0.5 * factor * (+ V[n + 1, 2] * f1meq1[n] + V[n + 1, 0] * f2meq1[n])
                        du2c[2, index] = - factor * V[n + 1, 1] * f3meq1[n]
                        du2s[2, index] = - factor * W[n + 1, 1] * f3meq1[n]
                    else:
                        du2c[0, index] = 0.5 * factor * (- V[n + 1, m + 1] * f1mgt1[index]
                                                         + V[n + 1, m - 1] * f2mgt1[index])
                        du2s[0, index] = 0.5 * factor * (- W[n + 1, m + 1] * f1mgt1[index]
                                                         + W[n + 1, m - 1] * f2mgt1[index])
                        du2c[1, index] = 0.5 * factor * (- W[n + 1, m + 1] * f1mgt1[index]
                                                         - W[n + 1, m - 1] * f2mgt1[index])
                        du2s[1, index] = 0.5 * factor * (+ V[n + 1, m + 1] * f1mgt1[index]
                                                         + V[n + 1, m - 1] * f2mgt1[index])
                        du2c[2, index] = - factor * V[n + 1, m] * f3mgt1[index]
                        du2s[2, index] = - factor * W[n + 1, m] * f3mgt1[index]

        return du2c, du2s


class ClibGravHar(AbsGravHar):

    def __init__(self):
        gravlib_filename = None
        super(ClibGravHar, self).__init__()
        '''# Attempt to find grav library to load'''
        if platform.system() == 'Windows':
            gravlib_filename = _glob.glob(str(_pathlib.Path(__file__).parent.parent.parent.joinpath('lib', 'GravHar.dll')))
        elif platform.system() == 'Linux':
            gravlib_filename = _glob.glob(str(_pathlib.Path(__file__).parent.parent.parent.joinpath('lib', 'GravHar.so')))
            # gravlib_filename = str(_pathlib.Path(__file__).parent.parent.parent.joinpath('lib', 'GravHar.so'))

        if len(gravlib_filename) == 0:
            raise ImportError('Unable to find the shared C library "GravHar".')
        self.__gravlib = cdll.LoadLibrary(gravlib_filename[0])

        '''
        interface def
        '''
        self.__gravlib.calVW.argtypes = [ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), c_int]
        self.__gravlib.preprocess.argtypes = [c_int]
        self.__gravlib.getPotential.argtypes = [c_int,
                                                ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                                ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
                                                ]
        self.__gravlib.getPotential.restype = c_double
        self.__gravlib.getAcc.argtypes = [c_int,
                                          ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                          ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                                          ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
                                          ]
        self.__gravlib.getDu2xyz.argtypes = [c_int,
                                             ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS'),
                                             ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS'),
                                             ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS')
                                             ]
        self.__gravlib.getDu2CS.argtypes = [c_int,
                                            c_int,
                                            ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS'),
                                            ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS'),
                                            ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS'),
                                            ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS'),
                                            ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS'),
                                            ndpointer(dtype=float, ndim=1, flags='C_CONTIGUOUS')
                                            ]

        self.__gravlib.setPar.argtypes = [c_double, c_double]

        pass

    def configure(self, FMConfig: ForceModelConfig):
        self._FMConfig = FMConfig
        self._GravHarConfig = self._FMConfig.GravHar()
        self._GravHarConfig.__dict__.update(self._FMConfig.GravHarConfig.copy())
        self._GM = self._GravHarConfig.GravHar_GM
        self._Radius = self._GravHarConfig.GravHar_Radius

        self.__gravlib.setPar(self._GM, self._Radius)
        return self

    def setPar(self, Cpot: np.ndarray, Spot: np.ndarray, Pos: np.ndarray):
        """
                Spherical harmonic coefficients in one-dimension (in terms of potential or geoid-height)
                :param Cpot: CPOT((N_max+1)*(N_max+2)/2),SPOT((N_max+1)*(N_max+2)/2)
                :param Spot:
                :param Pos: position in ITRS frame, a three dimension array
                :return:
                """
        # assert len(Pos) == 3
        assert len(Cpot) == len(Spot)
        L = len(Cpot)
        N = int(np.round(np.sqrt(2 * L))) - 1
        assert (N + 1) * (N + 2) / 2 == L

        if self._Nmax != N:
            self._Nmax = N
            self.__preprocess(N)

        self.__calVW(Pos)
        self._Cpot = Cpot
        self._Spot = Spot
        self._pos = Pos

        return self

    def __calVW(self, pos):

        # TYPE = c_double*len(pos)
        self.__gravlib.calVW(pos, self._Nmax)
        pass

    def __preprocess(self, Nmax):
        Nlimit = self.__gravlib.getNlimit()
        assert Nlimit > Nmax

        self.__gravlib.preprocess(Nmax)
        pass

    def get_potential(self):

        pot = self.__gravlib.getPotential(self._Nmax, self._Cpot, self._Spot)
        return pot

    def getAcceleration(self):

        acc = np.zeros(3)
        self.__gravlib.getAcc(self._Nmax, self._Cpot, self._Spot, acc)

        return acc.copy()

    # def getAccelerationGCRS(self, rm):
    #
    #
    #     acc = np.zeros(3)
    #     self.__gravlib.getAcc(self._Nmax, self._Cpot, self._Spot, acc)
    #
    #     acc = self.__time.AccITRS2GCRS(acc, self.__time.getRotationMatrix)
    #     acc = np.array(acc)
    #     return acc.copy()

    def getDaDp(self, degree_max: int = None, degree_min: int = None):
        if degree_max is None:
            degree_max = self._GravHarConfig.degree_max
        if degree_min is None:
            degree_min = self._GravHarConfig.degree_min
        assert 1 <= degree_min <= degree_max <= self._Nmax

        dim = int((degree_max + 1) * (degree_max + 2) / 2)
        dx2c = np.zeros(dim, dtype=np.float64)
        dy2c = np.zeros(dim, dtype=np.float64)
        dz2c = np.zeros(dim, dtype=np.float64)
        dx2s = np.zeros(dim, dtype=np.float64)
        dy2s = np.zeros(dim, dtype=np.float64)
        dz2s = np.zeros(dim, dtype=np.float64)

        self.__gravlib.getDu2CS(degree_min, degree_max, dx2c, dy2c, dz2c, dx2s, dy2s, dz2s)
        du2c = np.vstack((dx2c, dy2c))
        du2c = np.vstack((du2c, dz2c))
        du2s = np.vstack((dx2s, dy2s))
        du2s = np.vstack((du2s, dz2s))

        return du2c, du2s

    def getDaDx(self):

        du2xyz = np.zeros(9)
        self.__gravlib.getDu2xyz(self._Nmax, self._Cpot, self._Spot, du2xyz)

        return du2xyz.copy().reshape(3, 3)

