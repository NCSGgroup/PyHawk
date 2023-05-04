import numpy as np
from fractions import Fraction


class ODEcoeffGenerator:
    """
    Reference:
    1. "M. Berry, Healy, L. "Implementation of Gauss-Jackson Integration for
    Orbit Propagation". The Journal of Astronautical Sciences. V 52. N 3. Sept. 2004. pp 331-357.".
    2. Book: Satellite Orbits: Models, Methods and Applications, Chapter 4

    Definition of the 'difference form' and 'ordinate form' is found at ref-2.

    This is a generator for auto-computing coefficients for kinds of ODE solvers like below:
    1. Adams-bash predictor in difference form
    2. Adams-bash predictor in ordinate form
    3. Adams-Moulton corrector in difference form
    4. Adams-Moulton corrector in ordinate form
    5. Storm predictor in difference form
    6. Cowell corrector in difference form
    7. Storm, Cowell predictor/corrector in ordinate form is under development
    8. GaussJackon-Summed Adam predictor, corrector, mid-corrector in difference form
    9. GaussJackon-Summed Adam predictor, corrector, mid-corrector in ordinate form
    10. GaussJackson predictor, corrector, mid-corrector in difference form
    11. GaussJackson predictor, corrector, mid-corrector in ordinate form

    Note: All the coefficients are given in a list of fraction form. Check the results by running benchmark provided.
    """

    def __init__(self, order=8):
        """
        The order of ODE solver should be set by user, and the default is 8 that is consistent with the references
        :param order: the propagator should be preferred less than order 14 (to ease the programming) and
        greater than 4 (to gain computation precision)
        """
        self.__order = order
        assert 2 <= order <= 14
        pass

    @property
    def AdamsBash_Pred_Diff(self):
        dim = self.__order + 1
        coeff = []

        # Table 4.4 and Eq. 4.56 in reference 2, page 135
        coeff.append(Fraction(1, 1))
        for j in range(1, dim):
            r = 0
            for k in range(j):
                r += Fraction(1, j + 1 - k) * coeff[k]
            coeff.append(1 - r)

        return coeff

    @property
    def AdamsMoul_Corr_Diff(self):
        dim = self.__order + 1
        coeff = []

        # Table 4.6 and Eq. 4.64 in reference 2, page 135
        coeff.append(Fraction(1, 1))
        for j in range(1, dim):
            r = 0
            for k in range(j):
                r += Fraction(1, j + 1 - k) * coeff[k]
            coeff.append(- r)

        return coeff

    @property
    def AdamsBash_Pred_Ordinate(self):
        r_L = self.AdamsBash_Pred_Diff

        dim = self.__order + 1
        coeff = []

        # Table 4.6 and Eq. 4.64 in reference 2, page 135
        for j in range(1, dim):
            r = 0
            for l in range(self.__order - j, self.__order):
                r += r_L[l] * self.binomial(l, self.__order - j)
            coeff.append((-1) ** (self.__order - j) * r)

        return coeff

    @property
    def AdamsMoul_Corr_Ordinate(self):
        r_L = self.AdamsMoul_Corr_Diff

        dim = self.__order + 1
        coeff = []

        # Table 4.6 and Eq. 4.64 in reference 2, page 135
        for j in range(1, dim):
            r = 0
            for l in range(self.__order - j, self.__order):
                r += r_L[l] * self.binomial(l, self.__order - j)
            coeff.append((-1) ** (self.__order - j) * r)

        return coeff

    @property
    def Stoermer_pred_diff(self):
        # Table 4.8 and Eq. 4.89 and Eq. 4.90
        coeff = []

        r_L = self.AdamsMoul_Corr_Diff

        for j in range(self.__order + 1):
            coeff.append((1 - j) * r_L[j])

        return coeff

    @property
    def Cowell_corr_diff(self):
        # Table 4.8 and Eq. 4.92

        beta = self.Stoermer_pred_diff

        return self.diff(beta)

    def SummedAdam_diff(self, additional_form=False):
        """
        It is a summed form of Adams formula to compute velocity from acceleration in multiple-steps integration method
        like Gauss-Jackson
        See Eq.4.95 and Eq. 4.96 in ref 2; see ref 1 Table 3
        :param additional_form: if this is True, we set the first term of corrector to zero according to Eq 68, 69 ,70
        :return: an array-like in shape of (order+2)*(order+1)
        """

        order = self.__order
        self.__order += 1

        corrector = self.AdamsMoul_Corr_Diff
        corrector.pop(0)
        if additional_form:
            corrector[0] = 0

        predictor = self.AdamsBash_Pred_Diff
        predictor.pop(0)

        mid_corrector = []
        mid_corrector.append(self.diff(corrector))

        for i in range(order - 1):
            mid_corrector.append(self.diff(mid_corrector[i]))

        mid_corrector.reverse()

        # plus corrector and predictor
        mid_corrector.append(corrector)
        mid_corrector.append(predictor)

        self.__order = order
        return mid_corrector

    def GaussJackson_diff(self, additional_form=False):
        """
        It is a second sum of the Adam formulas; See Eq.4.93 and Eq. 4.94 in ref 2; see ref 1's Table 4
        :param additional_form: if this is True, we set the first term of corrector to zero according to Eq 68, 69 ,70
        :return: an array-like in shape of (order+2)*(order+1)
        """

        order = self.__order
        self.__order += 2

        corrector = self.Cowell_corr_diff
        corrector.pop(0)
        corrector.pop(0)
        if additional_form:
            corrector[0] = 0


        predictor = self.Stoermer_pred_diff
        predictor.pop(0)
        predictor.pop(0)

        mid_corrector = []
        mid_corrector.append(self.diff(corrector))

        for i in range(order - 1):
            mid_corrector.append(self.diff(mid_corrector[i]))

        mid_corrector.reverse()

        # plus corrector and predictor
        mid_corrector.append(corrector)
        mid_corrector.append(predictor)

        self.__order = order
        return mid_corrector

    @property
    def SummedAdam_ordinate(self):
        """
        See Table 5 in ref 1
        See Eq.65 and Eq. 66, pls note that the acceleration has a subscript of (n-m)
        :return:
        """
        coeff = []

        N = self.__order
        '''
        All the single-integration corrector and mid-corrector coefficients are computed this way; 
        however, the zero term is included in the coefficients for the predictor
        See Eq 68, 69 ,70 in ref 1.
        So we again define the first term of corrector to zero and calculate the corresponding modification
        of mid-corrector using Eq 63 in ref 1, like below
        '''
        diffcoeff = self.SummedAdam_diff(True)

        for k in range(N + 2):
            ordinate = []
            for m in range(N + 1):
                Znm = 0
                for i in range(m, N + 1):
                    Znm += diffcoeff[k][i] * self.binomial(i, m)

                Znm = Znm * (-1) ** m
                ordinate.append(Znm)

            ordinate.reverse()
            coeff.append(ordinate)

        return coeff

    @property
    def GaussJackson_ordinate(self):
        """
        See Table 5 in ref 1
        See Eq.65 and Eq. 66, pls note that the acceleration has a subscript of (n-m)
        :return:
        """
        coeff = []

        N = self.__order
        '''
        All the single-integration corrector and mid-corrector coefficients are computed this way; 
        however, the zero term is included in the coefficients for the predictor
        See Eq 68, 69 ,70 in ref 1.
        So we again define the first term of corrector to zero and calculate the corresponding modification
        of mid-corrector using Eq 63 in ref 1, like below
        !!!!Note: according to ref 1, the above change is unnecessary.
        '''
        diffcoeff = self.GaussJackson_diff(False)

        for k in range(N + 2):
            ordinate = []
            for m in range(N + 1):
                Znm = 0
                for i in range(m, N + 1):
                    Znm += diffcoeff[k][i] * self.binomial(i, m)

                Znm = Znm * (-1) ** m
                ordinate.append(Znm)

            ordinate.reverse()
            coeff.append(ordinate)

        return coeff

    @staticmethod
    def diff(inList: list):
        outList = []

        outList.append(inList[0])

        for i in range(1, len(inList)):
            outList.append(inList[i] - inList[i - 1])

        return outList

    @staticmethod
    def binomial(n=4, k=2):
        """
        see Eq. in reference 2
        :param n:
        :param k:
        :return:
        """
        if k == 0:
            return 1

        # numerator=func(n)
        x = n
        for i in range(1, k):
            x = x * (n - i)
            # print(i)
        numerator = x

        # fractional
        x = 1
        for i in range(1, k + 1):
            x = x * i
        denominator = x

        return Fraction(numerator, denominator)
