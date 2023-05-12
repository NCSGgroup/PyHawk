import numpy as np
from src.Preference.EnumType import ParaType
import src.Preference.backups.Pre_Parameterization as Pre_Parameterization


class Parameterization:
    """
    This class deals with the parametrization strategy of the orbit adjustment or gravity inversion process:
    To define the parameters to be estimated or solved.
    """
    def __init__(self):
        self.__reset()

    def __reset(self):
        self.__config = {}
        self.__APnum = None
        self.__sortCS = None
        pass

    def getInfo(self):
        pass

    def configure(self, **kwargs):
        self.__config[ParaType.TransitionMatrix] = kwargs.get("TransitionMatrix", Pre_Parameterization.TransitionMatrix)
        self.__config[ParaType.Accelerometer] = kwargs.get("Accelerometer", Pre_Parameterization.Accelerometer)
        self.__config[ParaType.StokesCoefficients] = kwargs.get("StokesCoefficients", Pre_Parameterization.StokesCoefficients)
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
                num = term.Parameter_Number
                if not term.AllisGlobal:
                    local_dm = np.dstack((local_dm, dm[:, :, index:index + num]))
                else:
                    global_dm = np.dstack((global_dm, dm[:, :, index:index + num]))
                index = index + num

        return rv, local_dm, global_dm

