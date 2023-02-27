import numpy as np
from src.Preference.EnumType import ParaType, ParaField



class Parametrization:
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

    def configure(self):
        pass

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
            rv = dm[0:3]
            index += 3

        local_dm = np.zeros((3, 0))
        global_dm = np.zeros((3, 0))

        for key in [ParaType.TransitionMatrix, ParaType.Accelerometer, ParaType.StokesCoefficients]:
            term = self.getInfo(key)
            if term.isRequired:
                num = term.ParaNum
                if term.field == ParaField.Local:
                    local_dm = np.hstack((local_dm, dm[index:index + num * 3].reshape((3, -1))))
                else:
                    global_dm = np.hstack((global_dm, dm[index:index + num * 3].reshape((3, -1))))
                index = index + num*3

        return rv, local_dm, global_dm