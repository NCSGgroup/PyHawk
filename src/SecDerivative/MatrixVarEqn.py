import numpy as np


class MatrixVarEqn:
    """
    This class mainly deals with the packing and unpacking of the variational equations.

    Formation of the variational Equations can refer to:
    1. Book: Satellite Orbits: Models, Methods and Applications, Chapter 7
    """

    def __init__(self):
        self.__ParNum = {'SensMat': 0,
                         'TransMat': 6,
                         'StateVec': 1}
        pass

    def setParNum(self, CScoef=100, OtherArg=10):
        self.__ParNum['SensMat'] = CScoef + OtherArg
        return self

    def getParNum(self, key: str):
        assert key in self.__ParNum.keys()
        return self.__ParNum[key]

    @staticmethod
    def VarEq1stOrder(PhiS: np.ndarray, ar, ap, acc,
                      isIncludeStateVec=False, isIncludeTransitionMatrix=False, isIncludeSensitivityMatrix=False):
        """
        This is the first order differential form of the variational Equation
        Reference 1, eq.(7.45)
        :param isIncludeSensitivityMatrix: The sensitivity matrix is included if true
        :param isIncludeTransitionMatrix: The transition matrix is included if true
        :param isIncludeStateVec: True, state vector is included. Vice the versa
        :param PhiS: StateVec [6*1] + Transition Matrix  [6*6] + Parameter Sensitivity matrix  [6*np] => one dimension
        :param ar: da/dr
        :param ap: da/dp
        :param acc: acceleration
        :return: right hand of the variational equation. [v, a, Phip, Sp]
        """
        PhiSp = None

        '''Unpack the PhiS'''
        index = 0
        if isIncludeStateVec:
            '''State vector [1*6]'''
            r = PhiS[0:3]
            v = PhiS[3:6]
            index += 6
            PhiSp = np.append(v, acc.flatten())

        if isIncludeTransitionMatrix:
            '''Phi, Transition Matrix, [6*6]'''
            Phi = PhiS[index: (index + 36)].reshape((6, 6))
            index += 6 * 6
            a = np.hstack((np.zeros((3, 3)), np.eye(3)))
            '''da/dr, da/dv'''
            b = np.hstack((ar, np.zeros((3, 3))))
            dfdy = np.vstack((a, b))
            Phip = np.matmul(dfdy, Phi)
            PhiSp = np.append(PhiSp, Phip.flatten())

        if isIncludeSensitivityMatrix:
            '''Sensitivity matrix, [6*np]'''
            S = PhiS[index:].reshape((6, -1))
            dimS = np.shape(S)
            dfdp = np.vstack(np.zeros((3, dimS[1])), ap)
            if not isIncludeTransitionMatrix:
                a = np.hstack((np.zeros((3, 3)), np.eye(3)))
                '''da/dr, da/dv'''
                b = np.hstack((ar, np.zeros((3, 3))))
                dfdy = np.vstack((a, b))
            Sp = np.matmul(dfdy, S) + dfdp
            PhiSp = np.append(PhiSp, Sp.flatten())

        return PhiSp

    # @staticmethod
    # def VarEq2ndOrder(PhirSr, PhivSv, ar, ap, acc,
    #                   isIncludeStateVec=False, isIncludeTransitionMatrix=False, isIncludeSensitivityMatrix=False):
    #     """
    #     Reference 1, eq.(7.48)
    #     This is the second order differential form of the variational Equation
    #     :param isIncludeSensitivityMatrix: The sensitivity matrix is included if true
    #     :param isIncludeTransitionMatrix: The transition matrix is included if true
    #     :param isIncludeStateVec: True, state vector is included. Vice the versa
    #     :param PhirSr: StateVec [3*1] + Transition Matrix  [3*6]
    #                    + Parameter Sensitivity matrix  [3*np] => one dimension
    #     :param PhivSv: time derivative of the PhirSr
    #     :param ar: da/dr
    #     :param ap: da/dp
    #     :param acc: acceleration
    #     :return: right hand of the variational equation. [a, Phip, Sp]
    #     """
    #     PhirSrp = None
    #     '''Unpack the PhiS'''
    #     index = 0
    #     if isIncludeStateVec:
    #         '''State vector [1*6]'''
    #         r = PhirSr[:, 0].copy()
    #         index += 3
    #         v = PhivSv[:, 0].copy()
    #         PhirSrp = np.array(acc).reshape((3, 1))
    #
    #     if isIncludeTransitionMatrix:
    #         '''Phi, Transition Matrix, [3*6]'''
    #         Phir = PhirSr[:, 1:7]
    #         index += 3 * 6
    #         Phirp = np.matmul(ar, Phir)
    #         PhirSrp = np.hstack((PhirSrp, Phirp))
    #
    #     if isIncludeSensitivityMatrix:
    #         '''Sensitivity matrix, [3*np]'''
    #         Sr = PhirSr[:, 7:]
    #         '''da/dp'''
    #         if isIncludeTransitionMatrix:
    #             Srp = np.matmul(ar, Sr) + ap
    #         else:
    #             Srp = ap
    #         PhirSrp = np.hstack((PhirSrp, Srp))
    #
    #     return PhirSrp

    @staticmethod
    def VarEq2ndOrder(PhirSr, PhivSv, ar, ap, acc,
                      isIncludeStateVec=False, isIncludeTransitionMatrix=False, isIncludeSensitivityMatrix=False):
        """
        Reference 1, eq.(7.48)
        This is the second order differential form of the variational Equation
        :param isIncludeSensitivityMatrix: The sensitivity matrix is included if true
        :param isIncludeTransitionMatrix: The transition matrix is included if true
        :param isIncludeStateVec: True, state vector is included. Vice the versa
        :param PhirSr: StateVec [3*1] + Transition Matrix  [3*6]
                       + Parameter Sensitivity matrix  [3*np] => one dimension
        :param PhivSv: time derivative of the PhirSr
        :param ar: da/dr
        :param ap: da/dp
        :param acc: acceleration
        :return: right hand of the variational equation. [a, Phip, Sp]
        """
        '''Unpack the PhiS'''

        PhirSrp = np.array(acc).reshape((3, 1))

        # if isIncludeTransitionMatrix:
        #     PhirS = np.matmul(ar, PhirSr[:, 1:])
        #     if isIncludeSensitivityMatrix:
        #         ap = np.hstack((np.zeros((3, 6)), ap))
        #         PhirSrp = np.hstack((PhirSrp, (PhirS + ap)))
        #     else:
        #         PhirSrp = np.hstack((PhirSrp, PhirS))
        # else:
        #     if isIncludeSensitivityMatrix:
        #         PhirSrp = np.hstack((PhirSrp, ap))
        if isIncludeTransitionMatrix:
            Phir = PhirSr[:, 1:7]
            PhirS = np.matmul(ar, Phir)
            PhirSrp = np.hstack((PhirSrp, PhirS))

        if isIncludeSensitivityMatrix:
            Sr = PhirSr[:, 7:]
            if isIncludeTransitionMatrix:
                Srp = np.matmul(ar, Sr) + ap
            else:
                Srp = ap
            PhirSrp = np.hstack((PhirSrp, Srp))

        return PhirSrp

    @staticmethod
    def getRVfromVarEq2ndOrder(PhirSr, PhivSv):
        """
        :param PhirSr: StateVec [3*1] + Transition Matrix  [3*6]
                       + Parameter Sensitivity matrix  [3*np] => one dimension
        :param PhivSv: time derivative of the PhirSr
        :return: pos [r], vel [v]
        """

        return PhirSr[:, 0].copy(), PhivSv[:, 0].copy()

    @staticmethod
    def getRVfromVarEq1stOrder(PhiS):
        """
        :param PhiS: StateVec [6*1] + Transition Matrix  [6*6] + Parameter Sensitivity matrix  [6*np] => one dimension
        :return: pos [r], vel [v]
        """

        return PhiS[0:3].copy(), PhiS[3:6].copy()

    @staticmethod
    def getVarEq1stIni(paraNum: int, isIncludeTransitionMat=False, isIncludeStateVec=False, stateVecIni=np.zeros(6)):
        """
        Initialization point for integration
        :param paraNum: Transition Matrix + Sensitivity Matrix = 6+np, which can be acquired from 'Parametrization.cls'
        :param isIncludeTransitionMat:
        :param isIncludeStateVec:
        :param stateVecIni: pos + vel
        :return: one-dimension
        """

        iniY = np.zeros(0)

        if isIncludeStateVec:
            iniY = np.append(iniY, stateVecIni)

        TransMat = np.zeros((6, paraNum))
        if isIncludeTransitionMat:
            TransMat[0, 0] = 1.
            TransMat[1, 1] = 1.
            TransMat[2, 2] = 1.
            TransMat[3, 3] = 1.
            TransMat[4, 4] = 1.
            TransMat[5, 5] = 1.

        iniY = np.append(iniY, TransMat.ravel())

        return iniY

    @staticmethod
    def getVarEq2ndIni(paraNum: int, isIncludeTransitionMat=False, isIncludeStateVec=True, stateVecIni=np.zeros(6)):
        """
        Initialization point for integration
        :param paraNum: Transition Matrix + Sensitivity Matrix = 6+np, which can be acquired from 'Parametrization.cls'
        :param isIncludeTransitionMat:
        :param isIncludeStateVec: This has to be set TRUE by default.
        :param stateVecIni: pos + vel
        :return: one-dimension
        """

        iniR = np.zeros(0)
        iniV = np.zeros(0)

        if isIncludeStateVec:
            iniR = np.array(stateVecIni[0:3]).reshape((3, 1))
            iniV = np.array(stateVecIni[3:6]).reshape((3, 1))

        # TransMat, TransMatV = np.zeros((3, paraNum)), np.zeros((3, paraNum))
        TransMat, TransMatV = np.zeros((3, 6)), np.zeros((3, 6))
        if isIncludeTransitionMat:
            TransMat[0, 0] = 1.
            TransMat[1, 1] = 1.
            TransMat[2, 2] = 1.
            TransMatV[0, 3] = 1.
            TransMatV[1, 4] = 1.
            TransMatV[2, 5] = 1.

        if paraNum > 6:
            iniR = np.hstack((iniR, TransMat))
            iniV = np.hstack((iniV, TransMatV))
            iniR = np.hstack((iniR, np.zeros((3, (paraNum - 6)))))
            iniV = np.hstack((iniV, np.zeros((3, (paraNum - 6)))))
        elif paraNum == 6:
            iniR = np.hstack((iniR, TransMat))
            iniV = np.hstack((iniV, TransMatV))
        else:
            iniR = np.hstack((iniR, np.zeros((3, paraNum))))
            iniV = np.hstack((iniV, np.zeros((3, paraNum))))

        return iniR, iniV

