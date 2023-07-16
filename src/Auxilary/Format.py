# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/7/15 16:54
# @File    : Format.py
# @Software: PyCharm
import os


class FormatWrite:
    def __init__(self):
        self.__fileDir = None
        self.__C = None
        self.__S = None
        self.__data = None
        self.__filename = None
        self.__fileFullPath = None
        self.product_type = 'gravity_field'
        self.modelname = None
        self.radius = '6.3781363000E+06'
        self.earth_gravity_constant = '3.9860044150E+14'
        self.max_degree = None
        self.norm = 'fully_normalized'
        self.tide_system = 'zero_tide'
        self.errors = 'formal'
        pass

    def configure(self, filedir, data, degree, c, s):
        self.__fileDir = filedir
        self.__data = data
        self.__C = c
        self.__S = s
        self.max_degree = degree
        self.modelname = 'GSM-2_' + self.__data[0] + '-' + self.__data[1] + '_GRAC_UTCSR_BA01_0600'
        self.__filename = 'HUST_HUST-Release-06_' + str(self.max_degree) +\
        'x' + str(self.max_degree) + '_unfiltered_GSM-2_' + self.__data[0] +\
        '-' + self.__data[1] + '_GRAC_HUST_BA01_0600.gfc'
        self.__fileDir()
        return self

    def __initFile(self):
        assert os.path.exists(self.__fileDir)

        self.__fileFullPath = os.path.join(self.__fileDir, self.__filename)

        if not os.path.exists(self.__fileFullPath):
            os.makedirs(self.__fileFullPath)
        return self

    def unfiltered(self):
        with open(self.__fileFullPath, 'w') as file:
            file.write('begin_of_head == == == == == == == == == == == == == == == == == == == == == == == == == =')
            file.write('%-31s%3s%-31s \n' % ('product_type', ':', self.product_type))
            file.write('%-31s%3s%-31s \n' % ('modelname', ':', self.modelname))
            file.write('%-31s%3s%-31s \n' % ('radius', ':', self.radius))
            file.write('%-31s%3s%-31s \n' % ('earth_gravity_constant', ':', self.earth_gravity_constant))
            file.write('%-31s%3s%-31s \n' % ('max_degree', ':', self.max_degree))
            file.write('%-31s%3s%-31s \n' % ('norm', ':', self.norm))
            file.write('%-31s%3s%-31s \n' % ('tide_system', ':', self.tide_system))
            file.write('%-31s%3s%-31s \n' % ('errors', ':', self.errors))

            # keys = list(self.__CS.Cnm.keys())
            # keys.sort()
            #
            # for key in keys:
            #     Cnm, Snm = self.__CS.Cnm[key], self.__CS.Snm[key]
            #     Nmax = self.__CS.maxDegree
            #     file.write('DATA SET %2i:   %s COEFFICIENTS FOR %s %s OF TYPE %s \n'
            #                % (keys.index(key), int((Nmax + 2) * (Nmax + 1) / 2), self.__CS.date, key.split('/')[0],
            #                   key.split('/')[1].lower()))
            #     self._mainContent(Cnm, Snm, Nmax, file)

        pass