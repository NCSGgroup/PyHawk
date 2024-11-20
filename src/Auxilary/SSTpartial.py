import numpy as np


class SSTpartial:
    """
    Partial of SST with respect to the state vectors (r, v).
    SST indicates the satellite-satellite tracking measurements, e.g., range, range-rate and range acceleration.
    Notice: This is a vectorized version (for matrix) !!!

    Reference:
    Kim,2002, PhD dissertation.
    """

    def __init__(self, Ra: np.ndarray, Va: np.ndarray, Rb: np.ndarray, Vb: np.ndarray):
        """
        Vectorized for N epochs.
        :param Ra: Position vector of sat A  [n*3]
        :param Va: Velocity vector of sat A [n*3]
        :param Rb: Position vector of sat B [n*3]
        :param Vb: Velocity vector of sat B [n*3]
        :return:
        """
        assert np.shape(Ra) == np.shape(Va) == np.shape(Rb) == np.shape(Vb)
        assert np.shape(Ra)[1] == 3

        self.__value = (Ra, Va, Rb, Vb)

        Ra, Va, Rb, Vb = self.__value
        diff_pos = Ra - Rb
        diff_vel = Va - Vb
        rhow = np.sqrt(diff_pos[:, 0] ** 2 + diff_pos[:, 1] ** 2 + diff_pos[:, 2] ** 2)
        rhow_dot = np.sum(diff_pos * diff_vel, 1) / rhow

        # e12 = diff_pos / rhow[:, None]
        # rhow_dot = diff_vel * e12
        self.Range = rhow[:, None]
        self.RangeRate = rhow_dot[:, None]
        pass

    def getPartial_Range(self):
        """
        get partial of range w.r.t state vectors.
        :return:
        """
        Ra, Va, Rb, Vb = self.__value
        diff_pos = Ra - Rb
        diff_vel = Va - Vb

        '''rhow w.r.t r1'''
        dd_r1 = diff_pos / self.Range
        '''rhow w.r.t r2'''
        dd_r2 = -dd_r1
        '''rhow w.r.t v1'''
        dd_v1 = np.zeros(np.shape(dd_r1))
        '''rhow w.r.t v2'''
        dd_v2 = np.zeros(np.shape(dd_r1))

        return dd_r1, dd_v1, dd_r2, dd_v2

    def getPartial_RangeRate(self):
        """
        get partial of range rate w.r.t state vectors
        :return:
        """
        Ra, Va, Rb, Vb = self.__value
        diff_pos = Ra - Rb
        diff_vel = Va - Vb

        '''rhow w.r.t r1'''
        dd_r1 = (diff_vel - diff_pos / self.Range * self.RangeRate) / self.Range
        '''rhow w.r.t r2'''
        dd_r2 = -dd_r1
        '''rhow w.r.t v1'''
        dd_v1 = diff_pos / self.Range
        '''rhow w.r.t v2'''
        dd_v2 = - dd_v1

        return dd_r1, dd_v1, dd_r2, dd_v2

    def getPartial_RangeAcc(self):
        return None
