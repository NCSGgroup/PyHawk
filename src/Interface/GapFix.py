import numpy as np
from scipy import interpolate
from src.Interface.L1b import L1b as Level_1B
from src.Preference.EnumType import Payload, SatID
import datetime
from Quaternion import Quat, Quaternion


class GapFix:
    def __init__(self, L1b: Level_1B):
        self.L1b = L1b
        self.__gap = {}
        self.__SCA_A = None
        self.__SCA_B = None
        self.__ACC_A = None
        self.__ACC_B = None
        pass

    def configure(self, **kwargs):
        self.__linear_mis_max = self.L1b.InterfaceConfig.gapfix_linear_mis_max
        self.__quadra_mis_max = self.L1b.InterfaceConfig.gapfix_quadra_mis_max
        self.__cubic_mis_max = self.L1b.InterfaceConfig.gapfix_cubic_mis_max
        self.__cubic_int = self.L1b.InterfaceConfig.gapfix_cubic_int
        self.__report_path = self.L1b.InterfacePathConfig.report_gap
        return self

    def fix_all(self):

        if not self.__checkOverLap():
            raise OSError

        SCAgap_A, self.__SCA_A = self.__fix(Payload.SCA, SatID.A)
        SCAgap_B, self.__SCA_B = self.__fix(Payload.SCA, SatID.B)
        ACCgap_A, self.__ACC_A = self.__fix(Payload.ACC, SatID.A)
        ACCgap_B, self.__ACC_B = self.__fix(Payload.ACC, SatID.B)

        KBR = self.L1b.getData(Payload.KBR.name, SatID.A)
        GNV_A = self.L1b.getData(Payload.GNV.name, SatID.A)
        GNV_B = self.L1b.getData(Payload.GNV.name, SatID.B)

        KBRgap = self.__detect(KBR, Payload.KBR)
        GNVgap_A = self.__detect(GNV_A, Payload.GNV)
        GNVgap_B = self.__detect(GNV_B, Payload.GNV)

        self.__gap['SCA-A'] = SCAgap_A
        self.__gap['SCA-B'] = SCAgap_B
        self.__gap['ACC-A'] = ACCgap_A
        self.__gap['ACC-B'] = ACCgap_B
        self.__gap['GNV-A'] = GNVgap_A
        self.__gap['GNV-B'] = GNVgap_B
        self.__gap['KBR'] = KBRgap

        return self

    def getData(self, type, sat: SatID):
        """

        :param type:
        :param sat:
        :return: a tuple (data, gapinfo)
        """
        if type == Payload.KBR.name:
            return self.L1b.getData(Payload.KBR.name, SatID.A), self.__gap['KBR']
        elif type == Payload.ACC.name:
            if sat == SatID.A:
                return self.__ACC_A, self.__gap['ACC-A']
            else:
                return self.__ACC_B, self.__gap['ACC-B']
        elif type == Payload.SCA.name:
            if sat == SatID.A:
                return self.__SCA_A, self.__gap['SCA-A']
            else:
                return self.__SCA_B, self.__gap['SCA-B']
        elif type == Payload.GNV.name:
            if sat == SatID.A:
                return self.L1b.getData(Payload.GNV.name, SatID.A), self.__gap['GNV-A']
            else:
                return self.L1b.getData(Payload.GNV.name, SatID.B), self.__gap['GNV-B']

    def getDate(self):
        return self.L1b.getDate()

    def makeReport(self):
        date = self.L1b.getDate()

        def leng(x):
            if x is None:
                return 0
            else:
                return len(x)

        self.__report_path = self.__report_path + '/' + date[0] + '_' + date[-1] + '.txt'
        with open(self.__report_path, 'w') as f:
            f.write('This ia a summary of all gaps that cannot be fixed.\n')
            f.write('Created by Yang Fan (yfan_cge@hust.edu.cn)\n')
            f.write('Record time: %s \n\n' % datetime.datetime.now())
            f.write('Gap number of ACC-A: %s\n' % leng(self.__gap['ACC-A']))
            f.write('Gap number of ACC-B: %s\n' % leng(self.__gap['ACC-B']))
            f.write('Gap number of SCA-A: %s\n' % leng(self.__gap['SCA-A']))
            f.write('Gap number of SCA_B: %s\n' % leng(self.__gap['SCA-B']))
            f.write('Gap number of GNV-A: %s\n' % leng(self.__gap['GNV-A']))
            f.write('Gap number of GNV-B: %s\n' % leng(self.__gap['GNV-B']))
            f.write('Gap number of KBR: %s\n\n' % leng(self.__gap['KBR']))

            f.write('-------------------Please see more details in below:---------------------- \n')
            for term in self.L1b.include_payloads:
                if term == Payload.KinematicOrbit.name:
                    '''do nothing for Kinematic Orbits'''
                    continue
                for sat in SatID:
                    if sat == SatID.B and (term == Payload.KBR.name or term == Payload.LRI.name):
                        continue

                    data, gapinfo = self.getData(term, sat)
                    f.write('%s + %s :\n' % (term, sat.name))

                    if leng(gapinfo) == 0:
                        f.write('0\n')
                    else:
                        for i in range(len(gapinfo)):
                            f.write('Gap %i: %s ===> %s ; NG: %i\n' % (i, data[gapinfo[i]['lr'], 0]
                                                                       , data[gapinfo[i]['rl'], 0], gapinfo[i]['ng']))
                    f.write('\n')

        return self

    def __fix(self, type: Payload, sat: SatID):

        data = self.L1b.getData(type.name, sat)
        gaplist = self.__detect(data, type)

        print('\n===============================================')
        print('Fix %s for %s ' % (type, sat))

        if gaplist is None:
            print('0 gap!!')
            return None, data

        iter = 0
        while True:
            print('Iteration %s : its gap number is %s' % (iter, len(gaplist)))
            iter += 1
            gaplist = self.__classify(gaplist, data, type)
            data = self.__fill(gaplist, data, type)
            gaplist2 = self.__detect(data, type)

            if not gaplist2:
                gaplist = gaplist2
                print('0 gap!!')
                return gaplist, data

            if len(gaplist2) == len(gaplist):
                gaplist = self.__classify(gaplist2, data, type)
                break
            else:
                gaplist = gaplist2

        print('Deeper fix begins working...')

        iter = 0
        while True:
            print('Iteration %s : its gap number is %s' % (iter, len(gaplist)))
            iter += 1

            data = self.__deeper(gaplist, data, type)
            gaplist2 = self.__detect(data, type)

            if not gaplist2:
                gaplist = gaplist2
                print('0 gap!!')
                break

            if len(gaplist2) == len(gaplist):
                break
            else:
                gaplist = gaplist2

        return gaplist, data

    def __checkOverLap(self):

        for term in self.L1b.include_payloads:
            for sat in SatID:
                if sat == SatID.B and (term == Payload.KBR.name or term == Payload.LRI.name):
                    continue

                data = self.L1b.getData(term, sat)
                time = data[:, 0]
                diff = time[1:] - time[0:-1]

                if (diff > 0).all():
                    pass
                else:
                    time2 = time[1:]
                    # logger.error('Data has overlaps for %s and %s, '
                    #              'please check GPS time: %s' % (term, sat, time2[diff <= 0]))
                    return False

        return True

    def __detect(self, data: np.ndarray, kind: Payload):

        # if kind == Payload.ACC:
        #     stepsize = 1
        # else:
        #     stepsize = 5

        stepsize = self.L1b.sr_payload[kind.name]

        gaplist = []

        time = data[:, 0]
        time2 = time[1:]
        time3 = time[0:-1]
        diff = time2 - time3

        temp = np.round(diff).astype(np.int64) == stepsize
        if temp.all():
            return None
        else:
            index = np.where(temp != True)[0]

            for i in range(len(index)):
                gapInfo = {}

                prev = i - 1
                next = i + 1

                if prev < 0:
                    gapInfo['ll'] = max(0, index[i] - self.__cubic_int + 1)
                else:
                    gapInfo['ll'] = max(index[prev] + 1, index[i] - self.__cubic_int + 1)

                gapInfo['lr'] = index[i]
                gapInfo['rl'] = index[i] + 1

                if next == len(index):
                    gapInfo['rr'] = min(len(time) - 1, index[i] + 1 + self.__cubic_int - 1)
                else:
                    gapInfo['rr'] = min(index[next], index[i] + 1 + self.__cubic_int - 1)

                gapInfo['ng'] = round((time[gapInfo['rl']] - time[gapInfo['lr']]) / stepsize - 1)
                gapInfo['nleft'] = gapInfo['lr'] - gapInfo['ll'] + 1
                gapInfo['nright'] = gapInfo['rr'] - gapInfo['rl'] + 1

                gaplist.append(gapInfo)

        return gaplist

    def __classify(self, gaplist: list, data, kind: Payload):

        time = data[:, 0]
        # if kind == Payload.ACC:
        #     stepsize = 1
        # else:
        #     stepsize = 5

        stepsize = self.L1b.sr_payload[kind.name]
        for gap in gaplist:

            '''Gap: the number of missing data'''
            ng = gap['ng']

            if gap['nleft'] == 1 or gap['nright'] == 1:
                gap['fill_m'] = 'linear'
                if ng <= self.__linear_mis_max:
                    gap['isBad'] = False
                else:
                    gap['isBad'] = True

            elif gap['nleft'] == 2 or gap['nright'] == 2:
                gap['fill_m'] = 'quadratic'
                if ng <= self.__quadra_mis_max:
                    gap['isBad'] = False
                else:
                    gap['isBad'] = True

            else:
                gap['fill_m'] = 'cubic'
                if ng <= self.__cubic_mis_max:
                    gap['isBad'] = False
                    if gap['rr'] - gap['ll'] < ng / 2:
                        gap['isBad'] = True
                else:
                    gap['isBad'] = True

        return gaplist

    def __fill(self, gaplist: list, data, kind: Payload):

        final = data[0:gaplist[0]['lr'] + 1, :]

        time = data[:, 0].astype(np.int64)

        # if kind == Payload.ACC:
        #     stepsize = 1
        # else:
        #     stepsize = 5
        stepsize = self.L1b.sr_payload[kind.name]

        for i in range(len(gaplist)):
            gap = gaplist[i]

            if not gap['isBad']:

                if gap['fill_m'] == 'linear':
                    x = data[gap['lr']:gap['rl'] + 1, 0]
                    y = data[gap['lr']:gap['rl'] + 1, 0:]
                    f = interpolate.interp1d(x, y.T)
                elif gap['fill_m'] == 'quadratic':
                    x = data[gap['lr'] - 1:gap['rl'] + 2, 0]
                    y = data[gap['lr'] - 1:gap['rl'] + 2, 0:]
                    f = interpolate.interp1d(x, y.T, kind='quadratic')
                else:
                    x = data[gap['ll']:gap['rr'] + 1, 0]
                    y = data[gap['ll']:gap['rr'] + 1, 0:]
                    f = interpolate.interp1d(x, y.T, kind='cubic')

                '''interpolation'''
                z_time = np.arange(time[gap['lr']] + stepsize, time[gap['rl']], stepsize)
                z_value = f(z_time)
                z_value[0] = np.round(z_value[0]).astype(np.float64)

                '''combination'''
                final = np.vstack((final, z_value.T))

            if i == len(gaplist) - 1:
                final = np.vstack((final, data[gap['rl']:, :]))
            else:
                gap_next = gaplist[i + 1]
                final = np.vstack((final, data[gap['rl']:gap_next['lr'] + 1, :]))

        return final

    def __deeper(self, gaplist: list, data, kind: Payload):
        """
        Deeper fix regards two gaps as a whole and interpolates the gap.
        :param gaplist:
        :param data:
        :param kind:
        :return:
        """

        final = data[0:gaplist[0]['lr'] + 1, :]
        time = data[:, 0].astype(np.int64)
        # if kind == Payload.ACC:
        #     stepsize = 1
        # else:
        #     stepsize = 5

        stepsize = self.L1b.sr_payload[kind.name]

        '''bug fix: gap list must not be shorter than 2'''
        if len(gaplist) <= 1:
            return data

        # if len(gaplist) % 2 == 0:
        #     length = len(gaplist) - 1
        # else:
        #     length = len(gaplist)

        for i in range(0, len(gaplist) - 1, 2):
            gap = gaplist[i]
            gap_next = gaplist[i + 1]

            ngAll = gap['ng'] + gap_next['ng']

            if ngAll <= self.__cubic_mis_max and gap['nleft'] > 2 and gap_next['nright'] > 2:
                x = data[gap['ll']:gap_next['rr'] + 1, 0]
                y = data[gap['ll']:gap_next['rr'] + 1, 0:]

                f = interpolate.interp1d(x, y.T, kind='cubic')

                '''interpolation'''
                z_time = np.arange(time[gap['lr']] + stepsize, time[gap_next['rl']], stepsize)
                z_value = f(z_time)
                z_value[0] = np.round(z_value[0]).astype(np.float64)

                '''combination'''
                final = np.vstack((final, z_value.T))
            else:
                final = np.vstack((final, data[gap['rl']:gap_next['lr'] + 1, :]))

            try:
                te = gaplist[i + 2]
            except IndexError:
                final = np.vstack((final, data[gap_next['rl']:, :]))
            else:
                if (len(gaplist) - 1) == (i + 2):
                    final = np.vstack((final, data[gap_next['rl']:, :]))
                else:
                    final = np.vstack((final, data[gap_next['rl']:gaplist[i + 2]['lr'] + 1, :]))

        return final
