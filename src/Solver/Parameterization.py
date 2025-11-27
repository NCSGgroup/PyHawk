import numpy as np
from src.Preference.EnumType import ParaType
from src.Preference.Pre_Parameterization import ParameterConfig


class Parameterization:
    """
    This class deals with the parametrization strategy of the orbit adjustment or gravity inversion process:
    To define the parameters to be estimated or solved.
    """
    def __init__(self):
        self._ParameterConfig = None
        self._TransitionMatrixConfig = None
        self._AccelerometerConfig = None
        self._StokesCoefficientsConfig = None
        self.nodes = None
        self.__reset()

    def __reset(self):
        self.__config = {}
        self.__APnum = None
        self.__sortCS = None
        pass

    def getInfo(self):
        pass

    def configure(self, parameterConfig: ParameterConfig, nodes):
        self._ParameterConfig = parameterConfig
        '''config TransitionMatrix'''
        self._TransitionMatrixConfig = self._ParameterConfig.TransitionMatrix()
        self._TransitionMatrixConfig.__dict__.update(self._ParameterConfig.TransitionMatrixConfig.copy())
        '''config AccelerometerConfig'''
        self._AccelerometerConfig = self._ParameterConfig.Accelerometer()
        self._AccelerometerConfig.__dict__.update(self._ParameterConfig.AccelerometerConfig.copy())
        '''config StokesCoefficientsConfig'''
        self._StokesCoefficientsConfig = self._ParameterConfig.StokesCoefficients()
        self._StokesCoefficientsConfig.__dict__.update(self._ParameterConfig.StokesCoefficientsConfig.copy())

        self.nodes = nodes
        self.__config[ParaType.TransitionMatrix] = self._TransitionMatrixConfig
        self.__config[ParaType.Accelerometer] = self._AccelerometerConfig
        self.__config[ParaType.StokesCoefficients] = self._StokesCoefficientsConfig
        return self

    def split_global_local(self, dm: np.ndarray, isIncludeStateVec=False):
        """
        for the 2-nd differential.
        split the design matrix according to the field of parameters.
        :param dm: Acquired from orbit integration directly. This could be dm_r, and also dm_v. [1-d]
        :param isIncludeStateVec: (x,y,z) or (vx, vy, vz)
        :return:
        """
        rv = None

        '''Unpack the PhiS'''
        index = 0
        if isIncludeStateVec:
            '''State vector [1*3]'''
            rv = dm[:, :, 0]
            index += 1

        local_dm = np.zeros((np.shape(dm)[0], np.shape(dm)[1], 0))
        global_dm = np.zeros((np.shape(dm)[0], np.shape(dm)[1], 0))

        for key in [ParaType.TransitionMatrix, ParaType.Accelerometer, ParaType.StokesCoefficients]:
            term = self.__config[key]
            if term.isRequired:
                if key is ParaType.Accelerometer:
                    if term.isScale:
                        num = 9
                        global_dm = np.dstack((global_dm, dm[:, :, index:index + num]))
                        index = index + num
                    num = self.nodes
                else:
                    num = term.Parameter_Number
                if not term.AllisGlobal:
                    local_dm = np.dstack((local_dm, dm[:, :, index:index + num]))
                else:
                    global_dm = np.dstack((global_dm, dm[:, :, index:index + num]))
                index = index + num

        return rv, local_dm, global_dm

