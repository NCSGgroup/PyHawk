# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/7/15 16:54
# @File    : Format.py
# @Software: PyCharm
import os
import time
from src.Auxilary.GeoMathKit import GeoMathKit


class FormatWrite:
    def __init__(self):
        self.__fileDir = None
        self.__C = None
        self.__S = None
        self.__sigmaC = None
        self.__sigmaS = None
        self.__data = None
        self.__filename = None
        self.__fileFullPath = None
        self.orderFirst = False
        self.product_type = 'gravity_field'
        self.modelname = None
        self.radius = '6.3781363000E+06'
        self.earth_gravity_constant = '3.9860044150E+14'
        self.max_degree = None
        self.norm = 'fully_normalized'
        self.tide_system = 'zero_tide'
        self.errors = 'formal'
        pass

    def configure(self, filedir, data, degree, c, s, sigmaC, sigmaS):
        self.__fileDir = filedir
        self.__data = data
        self.__C = c
        self.__S = s
        self.__sigmaC = sigmaC
        self.__sigmaS = sigmaS
        self.max_degree = degree
        self.modelname = 'GSM-2_' + self.__data[0] + '-' + self.__data[1] + '_GRAC_PyHawk_BA01_0600'
        self.__filename = 'HUST_HUST-Release-06_' + str(self.max_degree) +\
        'x' + str(self.max_degree) + '_unfiltered_GSM-2_' + self.__data[0]\
                          + self.__data[1] + '_GRAC_HUST_BA01_0600.gfc'
        self.__initFile()
        return self

    def __initFile(self):
        if not os.path.exists(self.__fileDir):
            os.makedirs(self.__fileDir)
        self.__fileFullPath = os.path.join(self.__fileDir, self.__filename)

        return self

    def unfiltered(self):
        with open(self.__fileFullPath, 'w') as file:

            file.write('********************************************************************************\n')
            file.write('model converted into ICGEM-format at: %s\n' % (time.strftime('%a %b %d %H:%M:%S %Y', time.localtime())))
            file.write('********************************************************************************\n')
            file.write('\n')
            file.write('begin_of_head ==================================================================\n')
            file.write('%-31s%-31s \n' % ('product_type', self.product_type))
            file.write('%-31s%-31s \n' % ('modelname', self.modelname))
            file.write('%-31s%-31s \n' % ('radius',  self.radius))
            file.write('%-31s%-31s \n' % ('earth_gravity_constant', self.earth_gravity_constant))
            file.write('%-31s%-31s \n' % ('max_degree',  self.max_degree))
            file.write('%-31s%-31s \n' % ('norm', self.norm))
            file.write('%-31s%-31s \n' % ('tide_system', self.tide_system))
            file.write('%-31s%-31s \n' % ('errors', self.errors))
            file.write('\n')
            file.write('%-10s%-7s%-10s%-20s%-15s%-15s%-31s \n' % ('key', 'L', 'M', 'C', 'S', 'sigma C', 'sigma S'))
            file.write('end_of_head ==========================================================================\n')

            self._mainContent(Cnm=self.__C, Snm=self.__S, SigmaC=self.__sigmaC,
                              SigmaS=self.__sigmaS, Nmax=self.max_degree, file=file)

        pass

    def _mainContent(self,Cnm, Snm, SigmaC, SigmaS, Nmax, file):
        Cnm = GeoMathKit.CS_1dTo2d(Cnm)
        Snm = GeoMathKit.CS_1dTo2d(Snm)
        SigmaC = GeoMathKit.CS_1dTo2d(SigmaC)
        SigmaS = GeoMathKit.CS_1dTo2d(SigmaS)
        if self.orderFirst:
            for i in range(Nmax + 1):
                for j in range(i + 1):
                    file.write('gfc %7i %6i  %+15.10E  %+15.10E %+14.4E  %+13.4E\n' %
                               (i, j, Cnm[i, j], Snm[i, j], SigmaC[i, j], SigmaS[i, j]))
        else:
            for j in range(Nmax + 1):
                for i in range(j, Nmax + 1):
                    file.write('gfc %7i %6i  %+15.10E  %+15.10E %+14.4E  %+13.4E\n' %
                               (i, j, Cnm[i, j], Snm[i, j], SigmaC[i, j], SigmaS[i, j]))

        pass


class FormatWrite_FO(FormatWrite):
    def __init__(self):
        super(FormatWrite_FO, self).__init__()

    def configure(self, filedir, data, degree, c, s, sigmaC, sigmaS):
        self.__fileDir = filedir
        self.__data = data
        self.__C = c
        self.__S = s
        self.__sigmaC = sigmaC
        self.__sigmaS = sigmaS
        self.max_degree = degree
        self.modelname = 'GSM-2_' + self.__data[0] + '-' + self.__data[1] + '_GRAC_UTCSR_BA01_0600'
        self.__filename = 'HUST_HUST-Release-06_' + str(self.max_degree) +\
        'x' + str(self.max_degree) + '_unfiltered_GSM-2_' + self.__data[0].split('-', '') \
                          + self.__data[1].split('-', '') + '_GRAC_HUST_BA01_0600.gfc'
        self.__initFile()
        return self