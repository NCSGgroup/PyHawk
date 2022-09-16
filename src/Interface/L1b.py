import src.Preference.Pre_Interface as Pre_Interface
from src.Auxilary.GeoMathKit import GeoMathKit
import numpy as np
from tqdm import trange
from src.Preference.EnumType import SatID, Payload, Mission
import os
import logging as lg
import struct
from src.Frame.Frame import Frame
import pathlib as pathlib


class L1b:

    def __init__(self):

        self._date = None
        self._ACC_A = None
        self._ACC_B = None
        self._SCA_A = None
        self._SCA_B = None
        self._KBR = None

        '''GNV orbit'''
        self.GNV_A = None
        self.GNV_B = None

        '''Kinematic Orbit'''
        self.KiOrb_A = None
        self.KiOrb_B = None

        pass

    def configure(self, **kwargs):
        self._path_ACC = kwargs.get('path_ACC', Pre_Interface.PathOfFiles.payload_ACC)
        self._path_KBR = kwargs.get('path_KBR', Pre_Interface.PathOfFiles.payload_KBR)
        self._path_SCA = kwargs.get('path_SCA', Pre_Interface.PathOfFiles.payload_SCA)
        self._path_GNV = kwargs.get('path_GNV', Pre_Interface.PathOfFiles.payload_GNV)
        self._path_LRI = kwargs.get('path_LRI', Pre_Interface.PathOfFiles.payload_LRI)

        include_payloads = kwargs.get('include_payloads', Pre_Interface.include_payloads)
        self.include_payloads = [key for key, value in include_payloads.items() if value]
        self.sr_payload = kwargs.get('sr_payload', Pre_Interface.sr_payload)

        '''isGCRS: tell the system that what's the frame of the GNV orbit. True--GCRS, False--ITRS'''
        self.is_GNV_GCRS = kwargs.get('is_GNV_GCRS', Pre_Interface.is_orbit_GCRS[Payload.GNV])
        return self

    def setDate(self, **kwargs):
        """
        :param begin_date:
        :param end_date:
        :param delete: delete the day that is not desired.
        :return:
        """
        date_span = kwargs.get('date_span', Pre_Interface.date_span)
        date_delete = kwargs.get('date_delete', Pre_Interface.date_delete)

        date_list = GeoMathKit.getEveryDay(*date_span)
        self._date = date_list
        for term in date_delete:
            if term in date_list:
                date_list.remove(term)
        return self

    def getDate(self):
        """
        :return: a list including all specified days
        """
        return self._date.copy()

    def read_single_sat(self, sat: SatID, readKBR: bool):
        date = self._date
        ACC = np.zeros((0, 4))
        SCA = np.zeros((0, 5))
        GNV = np.zeros((0, 7))
        KBR = np.zeros((0, 4))
        LRI = np.zeros((0, 4))

        for i in trange(len(date), desc='GRACE level 1b reading for %s' % sat,
                        disable=os.environ.get("DISABLE_TQDM", False)):
            day = date[i]

            '''ACC'''
            res = self._readACC(day, sat)
            if res is not None:
                ACC = np.vstack((ACC, res))

            '''SCA'''
            res2 = self._readSCA(day, sat)
            if res2 is not None:
                SCA = np.vstack((SCA, res2))

            '''GNV'''
            res3 = self._readGNV(day, sat)
            if res3 is not None:
                GNV = np.vstack((GNV, res3))

            '''KBR'''
            if not readKBR:
                continue
            res4 = self._readKBR(day)
            if res4 is not None:
                KBR = np.vstack((KBR, res4))

        if sat == SatID.A:
            self._ACC_A = ACC
            self._SCA_A = SCA
            self.GNV_A = GNV
        elif sat == SatID.B:
            self._ACC_B = ACC
            self._SCA_B = SCA
            self.GNV_B = GNV

        if readKBR:
            self._KBR = KBR

        pass

    def read_double_sat(self):
        self.read_single_sat(SatID.A, readKBR=True)
        self.read_single_sat(SatID.B, readKBR=True)
        pass

    def _readACC(self, date: str, sat: SatID):
        ACC = None
        return ACC

    def _readSCA(self, date: str, sat: SatID):
        SCA = None
        return SCA

    def _readKBR(self, date: str):
        KBR = None
        return KBR

    def _readGNV(self, date: str, sat: SatID):
        GNV = None
        return GNV

    def getData(self, type: Payload, sat: SatID):

        if type == Payload.KBR:
            return self._KBR
        elif type == Payload.ACC:
            if sat == SatID.A:
                return self._ACC_A
            else:
                return self._ACC_B
        elif type == Payload.SCA:
            if sat == SatID.A:
                return self._SCA_A
            else:
                return self._SCA_B
        elif type == Payload.GNV:
            if sat == SatID.A:
                return self.GNV_A
            else:
                return self.GNV_B
        elif type == Payload.KinematicOrbit:
            if sat == SatID.A:
                return self.KiOrb_A
            else:
                return self.KiOrb_B


class GRACE_OLD_RL02(L1b):
    def __init__(self):
        super(GRACE_OLD_RL02, self).__init__()
        assert Pre_Interface.mission is Mission.GRACE_RL02
        pass

    def _readACC(self, date: str, sat: SatID):
        path = self._path_ACC.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                       'ACC',
               'G'+ sat.name +'-OG-1B-ACCDAT+JPL-ACC1B_' + date + '_' + sat.name + '_02.dat'])
        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            print(path)
            return None

        NumOfAcc = None
        while True:
            r = f.readline()
            content = r.decode()
            # print(r.decode())
            if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
                NumOfAcc = int(content.split(':')[1])
            if content.strip() == 'END OF HEADER':
                break

        ACC = np.zeros((NumOfAcc, 4))

        for i in range(NumOfAcc):
            r = f.read(4 + 1 + 8 * 9 + 1)
            s = struct.unpack('>ic9dB', r)
            '''time'''
            ACC[i, 0] = s[0]
            '''X'''
            ACC[i, 1] = s[2]
            '''Y'''
            ACC[i, 2] = s[3]
            '''Z'''
            ACC[i, 3] = s[4]

        f.close()

        return ACC

    def _readSCA(self, date: str, sat: SatID):
        path = self._path_SCA.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                       'SCA',
               'G' + sat.name + '-OG-1B-SCAATT+JPL-SCA1B_' + date + '_' + sat.name + '_02.dat'])

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            print(path)
            return None

        NumOfSCA = None
        while True:
            r = f.readline()
            content = r.decode()
            # print(r.decode())
            if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
                NumOfSCA = int(content.split(':')[1])
            if content.strip() == 'END OF HEADER':
                break

        SCA = np.zeros((NumOfSCA, 5))

        for i in range(NumOfSCA):
            r = f.read(4 + 1 + 1 + 8 * 5 + 1)
            s = struct.unpack('>icb5dB', r)
            '''time'''
            SCA[i, 0] = s[0]
            '''cos(mu/2)'''
            SCA[i, 1] = s[3]
            '''I'''
            SCA[i, 2] = s[4]
            '''J'''
            SCA[i, 3] = s[5]
            '''K'''
            SCA[i, 4] = s[6]

        f.close()

        return SCA

    def _readKBR(self, date: str):
        path = self._path_KBR.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                         'KBR',
                                         'GX-OG-1B-KBRDAT+JPL-KBR1B_' + date + '_X_02.dat'])

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            print(path)
            return None

        NumOfKBR = None
        while True:
            r = f.readline()
            content = r.decode()
            # print(r.decode())
            if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
                NumOfKBR = int(content.split(':')[1])
            if content.strip() == 'END OF HEADER':
                break

        KBR = np.zeros((NumOfKBR, 4))

        for i in range(NumOfKBR):
            r = f.read(4 + 8 * 10 + 2 * 4 + 1)
            s = struct.unpack('>i10d4HB', r)
            '''time'''
            KBR[i, 0] = s[0]
            '''bias range + light_corr + ant_corr '''
            KBR[i, 1] = s[1] + s[5] + s[8]
            '''bias range-rate + light_corr + ant_corr '''
            KBR[i, 2] = s[2] + s[6] + s[9]
            '''bias range-acc + light_corr + ant_corr '''
            KBR[i, 3] = s[3] + s[7] + s[10]

        f.close()

        return KBR

    def _readGNV(self, date: str, sat: SatID):
        """
        Input is in ITRS frame, which has to be transferred to GCRS inertial frame; But probably the input orbit is gievn
        in GCRS. Take care !!!
        :param date:
        :param sat:
        :return: Orbit
        """
        st = date.split('-')
        IY, IM, ID = int(st[0]), int(st[1]), int(st[2])
        date_range = (round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 0, 0, 0))),
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.GNV]))))

        path = self._path_GNV.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                       'GNV',
               'G'+ sat.name +'-OG-1B-NAVSOL+JPL-GNV1B_' + date + '_' + sat.name + '_02.dat'])
        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            print(path)
            return None

        NumOfGNV = None
        while True:
            r = f.readline()
            content = r.decode()
            # print(r.decode())
            if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
                NumOfGNV = int(content.split(':')[1])
            if content.strip() == 'END OF HEADER':
                break

        # GNV = np.zeros((NumOfGNV, 7))
        GNV = []

        for i in range(NumOfGNV):
            """delete the data out of the range of given date"""

            r = f.read(4 + 1 + 1 + 8 * 12 + 1)
            s = struct.unpack('>icc12dB', r)

            if date_range[0] <= round(s[0]) <= date_range[1]:
                GNVd = np.zeros(7)
                pass
            else:
                continue

            '''time'''
            GNVd[0] = s[0]
            GNVd[1:4] = s[3:6]
            GNVd[4:7] = s[9:12]

            # if setGCRS:
            #     '''ITRS2GCRS'''
            #     rm = self.__fr.setTime(s[0]).getRotationMatrix
            #     '''pos'''
            #     GNVd[1:4] = Frame.PosITRS2GCRS(np.array(s[3:6]), rm)
            #     # GNVd[1:4] = s[3:6]
            #     '''vel'''
            #     GNVd[4:7] = Frame.VelITRS2GCRS(np.array(s[3:6]), np.array(s[9:12]), rm)
            #     # GNVd[4:7] = s[9:12]
            # else:
            #     GNVd[1:4] = s[3:6]
            #     GNVd[4:7] = s[9:12]

            GNV.append(GNVd)
            pass

        f.close()

        return np.array(GNV)


class GRACE_NEW_RL02(L1b):
    def __init__(self):
        super(GRACE_NEW_RL02, self).__init__()
        assert Pre_Interface.mission is Mission.GRACE_RL02
        pass

    def _readACC(self, date: str, sat: SatID):
        path = self._path_ACC.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                       'grace_1B_' + date + '_02.tar.gz_files',
                                         'ACC1B_' + date + '_' + sat.name + '_02.dat'])
        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            print(path)
            return None

        NumOfAcc = None
        while True:
            r = f.readline()
            content = r.decode()
            # print(r.decode())
            if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
                NumOfAcc = int(content.split(':')[1])
            if content.strip() == 'END OF HEADER':
                break

        ACC = np.zeros((NumOfAcc, 4))

        for i in range(NumOfAcc):
            r = f.read(4 + 1 + 8 * 9 + 1)
            s = struct.unpack('>ic9dB', r)
            '''time'''
            ACC[i, 0] = s[0]
            '''X'''
            ACC[i, 1] = s[2]
            '''Y'''
            ACC[i, 2] = s[3]
            '''Z'''
            ACC[i, 3] = s[4]

        f.close()

        return ACC

    def _readSCA(self, date: str, sat: SatID):
        path = self._path_ACC.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                       'grace_1B_' + date + '_02.tar.gz_files',
                                         'SCA1B_' + date + '_' + sat.name + '_02.dat'])

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            print(path)
            return None

        NumOfSCA = None
        while True:
            r = f.readline()
            content = r.decode()
            # print(r.decode())
            if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
                NumOfSCA = int(content.split(':')[1])
            if content.strip() == 'END OF HEADER':
                break

        SCA = np.zeros((NumOfSCA, 5))

        for i in range(NumOfSCA):
            r = f.read(4 + 1 + 1 + 8 * 5 + 1)
            s = struct.unpack('>icb5dB', r)
            '''time'''
            SCA[i, 0] = s[0]
            '''cos(mu/2)'''
            SCA[i, 1] = s[3]
            '''I'''
            SCA[i, 2] = s[4]
            '''J'''
            SCA[i, 3] = s[5]
            '''K'''
            SCA[i, 4] = s[6]

        f.close()

        return SCA

    def _readKBR(self, date: str):
        path = self._path_ACC.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                       'grace_1B_' + date + '_02.tar.gz_files',
                                         'KBR1B_' + date + '_X_02.dat'])

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            print(path)
            return None

        NumOfKBR = None
        while True:
            r = f.readline()
            content = r.decode()
            # print(r.decode())
            if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
                NumOfKBR = int(content.split(':')[1])
            if content.strip() == 'END OF HEADER':
                break

        KBR = np.zeros((NumOfKBR, 4))

        for i in range(NumOfKBR):
            r = f.read(4 + 8 * 10 + 2 * 4 + 1)
            s = struct.unpack('>i10d4HB', r)
            '''time'''
            KBR[i, 0] = s[0]
            '''bias range + light_corr + ant_corr '''
            KBR[i, 1] = s[1] + s[5] + s[8]
            '''bias range-rate + light_corr + ant_corr '''
            KBR[i, 2] = s[2] + s[6] + s[9]
            '''bias range-acc + light_corr + ant_corr '''
            KBR[i, 3] = s[3] + s[7] + s[10]

        f.close()

        return KBR

    def _readGNV(self, date: str, sat: SatID):
        """
        Input is in ITRS frame, which has to be transferred to GCRS inertial frame; But probably the input orbit is gievn
        in GCRS. Take care !!!
        :param date:
        :param sat:
        :return: Orbit
        """
        st = date.split('-')
        IY, IM, ID = int(st[0]), int(st[1]), int(st[2])
        date_range = (round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 0, 0, 0))),
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.GNV]))))

        path = self._path_ACC.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                       'grace_1B_' + date + '_02.tar.gz_files',
                                         'GNV1B_' + date + '_' + sat.name + '_02.dat'])
        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            print(path)
            return None

        NumOfGNV = None
        while True:
            r = f.readline()
            content = r.decode()
            # print(r.decode())
            if content.split(':')[0].strip() == 'NUMBER OF DATA RECORDS':
                NumOfGNV = int(content.split(':')[1])
            if content.strip() == 'END OF HEADER':
                break

        # GNV = np.zeros((NumOfGNV, 7))
        GNV = []

        for i in range(NumOfGNV):
            """delete the data out of the range of given date"""

            r = f.read(4 + 1 + 1 + 8 * 12 + 1)
            s = struct.unpack('>icc12dB', r)

            if date_range[0] <= round(s[0]) <= date_range[1]:
                GNVd = np.zeros(7)
                pass
            else:
                continue

            '''time'''
            GNVd[0] = s[0]
            GNVd[1:4] = s[3:6]
            GNVd[4:7] = s[9:12]

            # if setGCRS:
            #     '''ITRS2GCRS'''
            #     rm = self.__fr.setTime(s[0]).getRotationMatrix
            #     '''pos'''
            #     GNVd[1:4] = Frame.PosITRS2GCRS(np.array(s[3:6]), rm)
            #     # GNVd[1:4] = s[3:6]
            #     '''vel'''
            #     GNVd[4:7] = Frame.VelITRS2GCRS(np.array(s[3:6]), np.array(s[9:12]), rm)
            #     # GNVd[4:7] = s[9:12]
            # else:
            #     GNVd[1:4] = s[3:6]
            #     GNVd[4:7] = s[9:12]

            GNV.append(GNVd)
            pass

        f.close()

        return np.array(GNV)


class GRACE_FO_RL04(L1b):
    def __init__(self):
        super(GRACE_FO_RL04, self).__init__()
        assert Pre_Interface.mission is Mission.GRACE_FO_RL04
        pass

    def _readACC(self, date: str, sat: SatID):
        path = self._path_ACC.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                       'gracefo_1B_' + date + '_RL04.ascii.noLRI.tgz_files',
               'ACT1B_'+ date + '_' + chr(ord(sat.name) + 2) + '_04.txt'])
        try:
            f = open(path, 'r')
        except FileNotFoundError as e:
            print(path)
            return None

        NumOfAct = None
        while True:
            r = f.readline()
            if r.split(':')[0].strip() == 'num_records':
                NumOfAct = int(r.split(':')[1])
            if r.strip() == '# End of YAML header':
                break

        ACT = np.zeros((NumOfAct, 4))

        for i in range(NumOfAct):
            s = f.readline().split()
            '''time'''
            ACT[i, 0] = s[0]
            '''X'''
            ACT[i, 1] = s[2]
            '''Y'''
            ACT[i, 2] = s[3]
            '''Z'''
            ACT[i, 3] = s[4]

        f.close()
        return ACT

    def _readSCA(self, date: str, sat: SatID):
        path = self._path_SCA.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                         'gracefo_1B_' + date + '_RL04.ascii.noLRI.tgz_files',
                                         'SCA1B_' + date + '_' + chr(ord(sat.name) + 2) + '_04.txt'])
        try:
            f = open(path, 'r')
        except FileNotFoundError as e:
            print(path)
            return None

        NumOfSCA = None
        while True:
            r = f.readline()
            if r.split(':')[0].strip() == 'num_records':
                NumOfSCA = int(r.split(':')[1])
            if r.strip() == '# End of YAML header':
                break

        SCA = np.zeros((NumOfSCA, 5))

        for i in range(NumOfSCA):
            s = f.readline().split()
            '''time'''
            SCA[i, 0] = s[0]
            '''cos(mu/2)'''
            SCA[i, 1] = s[3]
            '''I'''
            SCA[i, 2] = s[4]
            '''J'''
            SCA[i, 3] = s[5]
            '''K'''
            SCA[i, 4] = s[6]

        f.close()

        return SCA

    def _readKBR(self, date: str):
        path = self._path_KBR.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                         'gracefo_1B_' + date + '_RL04.ascii.noLRI.tgz_files',
                                         'KBR1B_' + date + '_Y_04.txt'])

        try:
            f = open(path, 'r')
        except FileNotFoundError as e:
            lg.error(e)
            return None

        NumOfKBR = None
        while True:
            r = f.readline()
            if r.split(':')[0].strip() == 'num_records':
                NumOfKBR = int(r.split(':')[1])
            if r.strip() == '# End of YAML header':
                break

        KBR = np.zeros((NumOfKBR, 4))

        for i in range(NumOfKBR):
            s = f.readline().split()
            '''time'''
            KBR[i, 0] = s[0]
            '''bias range + light_corr + ant_corr '''
            KBR[i, 1] = float(s[1]) + float(s[5]) + float(s[8])
            '''bias range-rate + light_corr + ant_corr '''
            KBR[i, 2] = float(s[2]) + float(s[6]) + float(s[9])
            '''bias range-acc + light_corr + ant_corr '''
            KBR[i, 3] = float(s[3]) + float(s[7]) + float(s[10])
        f.close()

        return KBR

    def _readGNV(self, date: str, sat: SatID):
        """
        Input is in ITRS frame, which has to be transferred to GCRS inertial frame; But probably the input orbit is gievn
        in GCRS. Take care !!!
        :param date:
        :param sat:
        :return: Orbit
        """
        st = date.split('-')
        IY, IM, ID = int(st[0]), int(st[1]), int(st[2])
        date_range = (round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 0, 0, 0))),
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.GNV]))))

        path = self._path_GNV.joinpath(*[date.split('-')[0] + '-' + date.split('-')[1],
                                         'gracefo_1B_' + date + '_RL04.ascii.noLRI.tgz_files',
                                         'GNV1B_' + date + '_' + chr(ord(sat.name) + 2) + '_04.txt'])
        try:
            f = open(path, 'r')
        except FileNotFoundError as e:
            lg.error(e)
            return None

        NumOfGNV = None
        while True:
            r = f.readline()
            if r.split(':')[0].strip() == 'num_records':
                NumOfGNV = int(r.split(':')[1])
            if r.strip() == '# End of YAML header':
                break

        # GNV = np.zeros((NumOfGNV, 7))
        GNV = []

        for i in range(NumOfGNV):
            """delete the data out of the range of given date"""
            s = f.readline().split()
            if date_range[0] <= round(int(s[0])) <= date_range[1]:
                GNVd = np.zeros(7)
                pass
            else:
                continue

            '''time'''
            GNVd[0] = s[0]
            GNVd[1:4] = s[3:6]
            GNVd[4:7] = s[9:12]

            # if setGCRS:
            #     '''ITRS2GCRS''a'
            # GNId[0] = s[0]
            # fr = Frame(EOP().load(Setting.path.EOP).setPar(True))
            # rm = fr.setTime(int(s[0])).getRotationMatrix
            #     '''pos'''

            # GNId[1:4] = Frame.PosITRS2GCRS(np.array(s[3:6]).astype(float), rm)
            #     # GNVd[1:4] = s[3:6]
            #     '''vel'''
            # GNId[4:7] = Frame.VelITRS2GCRS(np.array(s[3:6]).astype(float), np.array(s[9:12]).astype(float), rm)
            #     # GNVd[4:7] = s[9:12]
            # else:
            #     GNVd[1:4] = s[3:6]
            #     GNVd[4:7] = s[9:12]

            # GNI.append(GNId)
            GNV.append(GNVd)

        f.close()
        return np.array(GNV)

