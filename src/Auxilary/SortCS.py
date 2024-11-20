"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/5/10 12:06
@Description:
"""

from src.Preference.EnumType import CSseq
import numpy as np
from src.Auxilary.GeoMathKit import GeoMathKit


class SortCS:
    """
    This class will sort the Stokes Coefficients in a given sequence in order to enhance the stability of
    normal equation. Several methods are given.
    Notice, the original sequence of C or S should obey the rule: 00 10 11 20 21 22 30 31 32 33 ....
    Notice, all is given in one-dimension
    """

    def __init__(self, degree_max, degree_min, method=CSseq.SortByOrder.name):

        if method == CSseq.Normal.name:
            self.__sort = self.__method1(degree_max, degree_min)
        elif method == CSseq.SortByOrder.name:
            self.__sort = self.__method2(degree_max, degree_min)
        elif method == CSseq.SortByDegree.name:
            self.__sort = self.__method3(degree_max, degree_min)
        else:
            self.__sort = self.__method4(degree_max, degree_min)

        self.__method = method
        self.__Nmax = degree_max
        self.__Nmin = degree_min

    def sort(self, C, S):
        """
        Sorting C, S in given law and combine them as a one-dimension vector
        :param C: one-dimension
        :param S: one-dimension
        :return:
        """
        indexC, indexS, indexCS_C, indexCS_S = self.__sort
        Nmax = self.__Nmax
        Nmin = self.__Nmin
        assert np.ndim(C) == np.ndim(S) == 1
        size = int((Nmax + 1) * (Nmax + 2) / 2)
        assert len(C) == len(S) == size

        CS = np.zeros((Nmax + 1) * (Nmax + 2) - Nmin * (Nmin + 1) - (Nmax - Nmin + 1), dtype=np.float64)

        CS[indexCS_C] = C[indexC]
        CS[indexCS_S] = S[indexS]

        return CS

    def invert(self, CS):
        """
        The inverted process of 'sort' function, aiming to factorize the CS sequence into C and S separately.
        :param CS: one-dimension
        :return:
        """
        indexC, indexS, indexCS_C, indexCS_S = self.__sort
        Nmax = self.__Nmax
        Nmin = self.__Nmin

        assert np.ndim(CS) == 1
        assert len(CS) == (Nmax + 1) * (Nmax + 2) - Nmin * (Nmin + 1) - (Nmax - Nmin + 1)

        CS = np.array(CS)
        size = int((Nmax + 1) * (Nmax + 2) / 2)
        C, S = np.zeros(size, dtype=np.float64), np.zeros(size, dtype=np.float64)

        C[indexC] = CS[indexCS_C]
        S[indexS] = CS[indexCS_S]

        return C, S

    @staticmethod
    def __method1(Nmax: int, Nmin=2):
        """
        Sort the C, S in a simple way [C, S]
        :param C: 00 10 11 20 21 22 30 31 32 33
        :param S: 00 10 11 20 21 22 30 31 32 33
        :return: CS sequence removing S00, S10, S20 .....
                e.g.: C00 C10 C20 ... S10 S11 S21 S22 S31 S32 ....
        """

        allsize_upper = (Nmax + 1) * (Nmax + 2) - Nmin * (Nmin + 1) - (Nmax - Nmin + 1)

        size_upper = int((Nmax + 1) * (Nmax + 2) / 2)
        size_lower = int(Nmin * (Nmin + 1) / 2)

        indexS = np.arange(size_upper)
        SS = np.ones(len(indexS))
        SS_new = GeoMathKit.Sx0(SS)
        SS_new[0: size_lower] = 0
        index_new = np.nonzero(SS_new)

        indexC = np.arange(size_lower, size_upper)
        indexS = index_new
        indexCS_C = np.arange(size_upper - size_lower)
        indexCS_S = np.arange(size_upper - size_lower, allsize_upper)

        return indexC, indexS, indexCS_C, indexCS_S

    @staticmethod
    def __method2(Nmax: int, Nmin=2):
        """
        Sort the C, S in Colombo sequence, sorted by order
        :param C: 00 10 11 20 21 22 30 31 32 33
        :param S: 00 10 11 20 21 22 30 31 32 33
        :return: C20 C30 C40 C50 C21 C31 C41 C51 S21 S31 S41 S51 ....
        """
        indexC = []
        indexS = []
        indexCS_C = []
        indexCS_S = []
        i = -1
        for m in range(0, Nmax + 1):

            '''cycle for C'''
            for l in range(m, Nmax + 1):

                if l >= Nmin:
                    a = int((l + 1) * l / 2 + m)
                    indexC.append(a)
                    i += 1
                    indexCS_C.append(i)

                pass

            '''cycle for S'''
            for l in range(m, Nmax + 1):

                if l >= Nmin and m != 0:
                    b = int((l + 1) * l / 2 + m)
                    indexS.append(b)
                    i += 1
                    indexCS_S.append(i)
                pass

        return np.array(indexC), np.array(indexS), np.array(indexCS_C), np.array(indexCS_S)

    def __method3(self, Nmax: int, Nmin=2):
        """
        Sort the C, S by degree
        :param C: 00 10 11 20 21 22 30 31 32 33
        :param S: 00 10 11 20 21 22 30 31 32 33
        :return: C20 C21 C22 S21 S22 C30 C31 C32 C33 S31 S32 S33 ....
        """
        indexC = []
        indexS = []
        indexCS_C = []
        indexCS_S = []

        i = -1
        for l in range(Nmin, Nmax + 1):

            '''cycle for C'''
            for m in range(0, l + 1):
                a = int((l + 1) * l / 2 + m)
                indexC.append(a)
                i += 1
                indexCS_C.append(i)

            '''cycle for S'''
            for m in range(1, l + 1):
                if l >= Nmin:
                    b = int((l + 1) * l / 2 + m)
                    indexS.append(b)
                    i += 1
                    indexCS_S.append(i)

                pass

        return np.array(indexC), np.array(indexS), np.array(indexCS_C), np.array(indexCS_S)

    def __method4(self, Nmax: int, Nmin=2):
        """
        Not available
        """

        return None, None, None, None


