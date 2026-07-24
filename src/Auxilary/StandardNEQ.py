"""
@Description:
    Save PyHawk monthly normal equations in an ITSG-like SINEX format.

    The main PyHawk solver keeps its own HDF5 normal equation. This helper only
    writes an extra standard NEQ file after monthly accumulation. It reorders
    gravity parameters to the ITSG degree-major sequence:
        C20, C21, S21, C22, S22, C30, C31, S31, ...
"""

import datetime
import os

import numpy as np

from src.Preference.EnumType import CSseq


class StandardNEQ:
    def __init__(self):
        self.__fileDir = None
        self.__filename = None
        self.__fileFullPath = None
        self.__dateSpan = None
        self.__degreeMax = None
        self.__degreeMin = None
        self.__sortMethod = None
        self.__scaleParameterCount = 0
        self.__productName = 'Hawk-NEQ1C'
        self.__aprioriModel = None
        pass

    def configure(self, filedir, date_span, degree_max, degree_min=2,
                  sort_method=CSseq.SortByOrder.name, scale_parameter_count=0,
                  filename=None, product_name='Hawk-NEQ1C',
                  apriori_model=None):
        self.__dateSpan = date_span
        self.__fileDir = os.path.join(
            str(filedir), str(self.__dateSpan[0])[0:4]
        )
        self.__degreeMax = int(degree_max)
        self.__degreeMin = int(degree_min)
        self.__sortMethod = sort_method
        self.__scaleParameterCount = int(scale_parameter_count)
        self.__productName = product_name
        self.__aprioriModel = apriori_model

        if filename is None:
            filename = self.__make_filename()
        self.__filename = filename
        self.__fileFullPath = os.path.join(self.__fileDir, self.__filename)
        return self

    def save(self, NMatrix, LMatrix):
        NMatrix, LMatrix = self.__format_input(NMatrix=NMatrix, LMatrix=LMatrix)
        NMatrix, LMatrix = self.__keep_gravity_only(NMatrix=NMatrix, LMatrix=LMatrix)
        NMatrix, LMatrix, parameters, source_index = self.__reorder_to_standard(
            NMatrix=NMatrix,
            LMatrix=LMatrix
        )

        if not os.path.exists(self.__fileDir):
            os.makedirs(self.__fileDir, exist_ok=True)

        with open(self.__fileFullPath, 'w') as file:
            self.__write_header(file=file, parameter_number=len(parameters))
            self.__write_reference(file=file)
            self.__write_comment(file=file)
            self.__write_statistics(file=file, parameter_number=len(parameters))
            self.__write_vector(file=file, LMatrix=LMatrix, parameters=parameters)
            self.__write_matrix(file=file, NMatrix=NMatrix)
            file.write('%ENDSNX\n')

        return self.__fileFullPath

    def __make_filename(self):
        date_tag = self.__date_tag()
        return '%s_n%d_%s.snx' % (self.__productName, self.__degreeMax, date_tag)

    def __date_tag(self):
        start = self.__dateSpan[0]
        end = self.__dateSpan[1]
        if start[0:7] == end[0:7]:
            return start[0:7]
        return start + '_' + end

    def __format_input(self, NMatrix, LMatrix):
        NMatrix = np.asarray(NMatrix, dtype=np.float64)
        LMatrix = np.asarray(LMatrix, dtype=np.float64).reshape((-1, 1))

        if np.ndim(NMatrix) != 2 or NMatrix.shape[0] != NMatrix.shape[1]:
            raise ValueError('NMatrix should be a square matrix')
        if NMatrix.shape[0] != LMatrix.shape[0]:
            raise ValueError('NMatrix and LMatrix size mismatch')

        return NMatrix, LMatrix

    def __keep_gravity_only(self, NMatrix, LMatrix):
        if self.__scaleParameterCount <= 0:
            return NMatrix, LMatrix

        num = self.__scaleParameterCount
        if num >= NMatrix.shape[0]:
            raise ValueError('scale parameter count is larger than NEQ size')

        N_ss = NMatrix[0:num, 0:num]
        N_sg = NMatrix[0:num, num:]
        N_gs = NMatrix[num:, 0:num]
        N_gg = NMatrix[num:, num:]
        l_s = LMatrix[0:num]
        l_g = LMatrix[num:]

        M_n = np.linalg.lstsq(N_ss, N_sg, rcond=None)[0]
        M_l = np.linalg.lstsq(N_ss, l_s, rcond=None)[0]

        N_out = N_gg - N_gs @ M_n
        l_out = l_g - N_gs @ M_l
        return N_out, l_out

    def __reorder_to_standard(self, NMatrix, LMatrix):
        source_parameters = self.__get_source_parameters()
        standard_parameters = self.__get_standard_parameters()

        if len(source_parameters) != NMatrix.shape[0]:
            raise ValueError(
                'parameter number mismatch: %d from config, %d in NEQ' %
                (len(source_parameters), NMatrix.shape[0])
            )

        source_map = {}
        for i, item in enumerate(source_parameters):
            source_map[item] = i

        index = []
        source_index = []
        for item in standard_parameters:
            if item not in source_map:
                raise ValueError('missing parameter in source NEQ: %s%d%d' % item)
            source_i = source_map[item]
            index.append(source_i)
            source_index.append(source_i + 1)

        index = np.array(index, dtype=np.int64)
        N_out = NMatrix[np.ix_(index, index)]
        l_out = LMatrix[index]
        return N_out, l_out, standard_parameters, source_index

    def __get_source_parameters(self):
        method = self.__sortMethod
        if method == CSseq.Normal.name:
            return self.__get_normal_parameters()
        elif method == CSseq.SortByOrder.name:
            return self.__get_order_parameters()
        elif method == CSseq.SortByDegree.name:
            return self.__get_degree_parameters()
        else:
            raise ValueError('unsupported CS sort method: %s' % method)

    def __get_standard_parameters(self):
        parameters = []
        for n in range(self.__degreeMin, self.__degreeMax + 1):
            parameters.append(('CN', n, 0))
            for m in range(1, n + 1):
                parameters.append(('CN', n, m))
                parameters.append(('SN', n, m))
        return parameters

    def __get_order_parameters(self):
        parameters = []
        for m in range(0, self.__degreeMax + 1):
            for n in range(max(m, self.__degreeMin), self.__degreeMax + 1):
                parameters.append(('CN', n, m))
            if m != 0:
                for n in range(max(m, self.__degreeMin), self.__degreeMax + 1):
                    parameters.append(('SN', n, m))
        return parameters

    def __get_degree_parameters(self):
        parameters = []
        for n in range(self.__degreeMin, self.__degreeMax + 1):
            for m in range(0, n + 1):
                parameters.append(('CN', n, m))
            for m in range(1, n + 1):
                parameters.append(('SN', n, m))
        return parameters

    def __get_normal_parameters(self):
        parameters = []
        for n in range(self.__degreeMin, self.__degreeMax + 1):
            for m in range(0, n + 1):
                parameters.append(('CN', n, m))
        for n in range(self.__degreeMin, self.__degreeMax + 1):
            for m in range(1, n + 1):
                parameters.append(('SN', n, m))
        return parameters

    def __write_header(self, file, parameter_number):
        now = datetime.datetime.utcnow()
        start = self.__sinex_epoch(self.__dateSpan[0])
        end = self.__sinex_epoch(self.__dateSpan[1])
        create = '%02d:%03d:%05d' % (
            now.year % 100,
            int(now.strftime('%j')),
            now.hour * 3600 + now.minute * 60 + now.second
        )
        file.write('%%=SNX 2.02 HUST %s HUST %s %s C %05d 2            2\n' %
                   (create, start, end, parameter_number))
        file.write('*-------------------------------------------------------------------------------\n')

    def __write_reference(self, file):
        file.write('+FILE/REFERENCE\n')
        file.write('*INFO_TYPE_________ INFO________________________________________________________\n')
        file.write(' DESCRIPTION        %s\n' % self.__filename.replace('.snx', ''))
        file.write(' SOFTWARE           PyHawk\n')
        file.write('-FILE/REFERENCE\n')
        file.write('*-------------------------------------------------------------------------------\n')

    def __write_comment(self, file):
        file.write('+FILE/COMMENT\n')
        file.write(' Hawk monthly normal equation standardized in ITSG-like SINEX layout\n')
        file.write(' modelname              %s\n' % self.__filename.replace('.snx', ''))
        file.write(' product_type           gravity_field\n')
        file.write(' max_degree             %d\n' % self.__degreeMax)
        file.write(' norm                   fully_normalized\n')
        file.write(' tide_system            zero_tide\n')
        file.write(' source_sort_method     %s\n' % self.__sortMethod)
        file.write(' standard_sort_method   ITSG_degree_major\n')
        if self.__aprioriModel is not None:
            file.write(' apriori_model_name     %s\n' % self.__aprioriModel.modelname)
        file.write('-FILE/COMMENT\n')
        file.write('*-------------------------------------------------------------------------------\n')

    def __write_statistics(self, file, parameter_number):
        file.write('+SOLUTION/STATISTICS\n')
        file.write('*____STATISTICAL_PARAMETER_____ _______VALUE(S)_______\n')
        file.write(' NUMBER OF UNKNOWNS                         %10d\n' % parameter_number)
        file.write(' SCALE PARAMETERS ELIMINATED                %10d\n' % self.__scaleParameterCount)
        file.write('-SOLUTION/STATISTICS\n')
        file.write('*-------------------------------------------------------------------------------\n')

    def __write_vector(self, file, LMatrix, parameters):
        epoch = self.__mid_epoch()
        file.write('+SOLUTION/NORMAL_EQUATION_VECTOR\n')
        file.write('*INDEX _TYPE_ CODE PT SOLN _REF_EPOCH__ UNIT S ___RIGHT_HAND_SIDE___\n')
        for i, item in enumerate(parameters):
            kind, degree, order = item
            file.write('%6d %-2s %8d -- %4d %s ---- 2 % .14e\n' %
                       (i + 1, kind, degree, order, epoch, LMatrix[i, 0]))
        file.write('-SOLUTION/NORMAL_EQUATION_VECTOR\n')
        file.write('*-------------------------------------------------------------------------------\n')

    def __write_matrix(self, file, NMatrix):
        size = NMatrix.shape[0]
        file.write('+SOLUTION/NORMAL_EQUATION_MATRIX U\n')
        file.write('*PARA1 PARA2 _______PARA2+0_______ _______PARA2+1_______ _______PARA2+2_______\n')
        for i in range(size):
            for j in range(i, size, 3):
                values = NMatrix[i, j:min(j + 3, size)]
                line = '%6d %5d' % (i + 1, j + 1)
                for value in values:
                    line += ' % .14e' % value
                file.write(line + '\n')
        file.write('-SOLUTION/NORMAL_EQUATION_MATRIX\n')

    def __sinex_epoch(self, date_str):
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        return '%02d:%03d:00000' % (date.year % 100, int(date.strftime('%j')))

    def __mid_epoch(self):
        date1 = datetime.datetime.strptime(self.__dateSpan[0], '%Y-%m-%d')
        date2 = datetime.datetime.strptime(self.__dateSpan[1], '%Y-%m-%d')
        mid = date1 + (date2 - date1) / 2
        return '%02d:%03d:00000' % (mid.year % 100, int(mid.strftime('%j')))
