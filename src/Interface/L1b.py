from src.Auxilary.GeoMathKit import GeoMathKit
import numpy as np
from tqdm import trange
from src.Preference.EnumType import SatID, Payload, Mission
import os
import logging as lg
import struct
from src.Frame.Frame import Frame
from src.Preference.Pre_Interface import InterfaceConfig
import pathlib as pathlib
import sys


class L1b:

    def __init__(self):
        self.dir = pathlib.Path()
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

        self.InterfaceConfig = None
        self.InterfacePathConfig = None
        self.OrbitConfig = None
        pass

    def configure(self, InterfaceConfig:InterfaceConfig, **kwargs):
        '''config Interface'''
        self.InterfaceConfig = InterfaceConfig
        self.InterfacePathConfig = InterfaceConfig.PathOfFiles(self.InterfaceConfig)
        self.InterfacePathConfig.__dict__.update(self.InterfaceConfig.PathOfFilesConfig.copy())
        self.OrbitConfig = InterfaceConfig.Orbit()
        self.OrbitConfig.__dict__.update(InterfaceConfig.OrbitConfig.copy())
        self._path_ACC = self.InterfacePathConfig.payload_ACC
        self._path_KBR = self.InterfacePathConfig.payload_KBR
        self._path_SCA = self.InterfacePathConfig.payload_SCA
        self._path_GNV = self.InterfacePathConfig.payload_GNV
        self._path_LRI = self.InterfacePathConfig.payload_LRI

        include_payloads = self.InterfaceConfig.include_payloads
        self.include_payloads = [key for key, value in include_payloads.items() if value]
        self.sr_payload = self.InterfaceConfig.sr_payload

        '''isGCRS: tell the system that what's the frame of the GNV orbit. True--GCRS, False--ITRS'''
        self.is_GNV_GCRS = self.InterfaceConfig.is_orbit_GCRS[Payload.GNV.name]
        return self

    def setDate(self, **kwargs):
        """
        :param begin_date:
        :param end_date:
        :param delete: delete the day that is not desired.
        :return:
        """
        date_span = self.InterfaceConfig.date_span
        date_delete = self.InterfaceConfig.date_delete

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

        SCA = self._unique(SCA)
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

    def getData(self, type, sat: SatID):

        if type == Payload.KBR.name:
            return self._KBR
        elif type == Payload.ACC.name:
            if sat == SatID.A:
                return self._ACC_A
            else:
                return self._ACC_B
        elif type == Payload.SCA.name:
            if sat == SatID.A:
                return self._SCA_A
            else:
                return self._SCA_B
        elif type == Payload.GNV.name:
            if sat == SatID.A:
                return self.GNV_A
            else:
                return self.GNV_B
        elif type == Payload.KinematicOrbit.name:
            if sat == SatID.A:
                return self.KiOrb_A
            else:
                return self.KiOrb_B

    def _unique(self, data):
        time = data[:, 0]
        index = np.unique(time, return_index=True)[1]
        return data[index, :]


class GRACE_OLD_RL02(L1b):
    def __init__(self):
        super(GRACE_OLD_RL02, self).__init__()
        # assert self._InterfaceConfig.mission == Mission.GRACE_RL02.name
        pass

    def _readACC(self, date: str, sat: SatID):
        path = str(self.dir.joinpath(self._path_ACC, date.split('-')[0] + '-' + date.split('-')[1],
                     'ACC', 'G' + sat.name + '-OG-1B-ACCDAT+JPL-ACC1B_' + date + '_' + sat.name + '_02.dat'))
        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
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
        path = str(self.dir.joinpath(self._path_SCA, date.split('-')[0] + '-' + date.split('-')[1],
                         'SCA', 'G' + sat.name + '-OG-1B-SCAATT+JPL-SCA1B_' + date + '_' + sat.name + '_02.dat'))
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
        path = str(self.dir.joinpath(self._path_KBR, date.split('-')[0] + '-' + date.split('-')[1],
                              'KBR', 'GX-OG-1B-KBRDAT+JPL-KBR1B_' + date + '_X_02.dat'))

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
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.GNV.name]))))

        path = str(self.dir.joinpath(self._path_GNV, date.split('-')[0] + '-' + date.split('-')[1],
                              'GNV', 'G' + sat.name + '-OG-1B-NAVSOL+JPL-GNV1B_' + date + '_' + sat.name + '_02.dat'))

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
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
        # assert self._InterfaceConfig.mission == Mission.GRACE_RL02.name
        pass

    def _readACC(self, date: str, sat: SatID):
        path = str(self.dir.joinpath(self._path_ACC, date.split('-')[0] + '-' + date.split('-')[1],
                              'grace_1B_' + date + '_02', 'ACC1B_' + date + '_' + sat.name + '_02.dat'))
        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # print(path)
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
        path = str(self.dir.joinpath(self._path_SCA, date.split('-')[0] + '-' + date.split('-')[1],
                              'grace_1B_' + date + '_02', 'SCA1B_' + date + '_' + sat.name + '_02.dat'))

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
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
        path = str(self.dir.joinpath(self._path_KBR, date.split('-')[0] + '-' + date.split('-')[1],
                              'grace_1B_' + date + '_02', 'KBR1B_' + date + '_X_02.dat'))

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
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
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.GNV.name]))))

        path = str(self.dir.joinpath(self._path_GNV, date.split('-')[0] + '-' + date.split('-')[1],
                              'grace_1B_' + date + '_02', 'GNV1B_' + date + '_' + sat.name + '_02.dat'))

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
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


class GRACE_RL03(L1b):
    def __init__(self):
        super(GRACE_RL03, self).__init__()
        # assert self._InterfaceConfig.mission == Mission.GRACE_RL02.name
        pass

    def _readACC(self, date: str, sat: SatID):
        path = str(self.dir.joinpath(self._path_ACC, date.split('-')[0] + '-' + date.split('-')[1],
                              'grace_1B_' + date + '_02', 'ACC1B_' + date + '_' + sat.name + '_02.dat'))
        st = date.split('-')
        IY, IM, ID = int(st[0]), int(st[1]), int(st[2])
        date_range = (round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 0, 0, 0))),
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.ACC.name]))))
        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # print(path)
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

        if NumOfAcc == 0:
            return None

        # ACC = np.zeros((NumOfAcc, 4))
        ACC = []
        for i in range(NumOfAcc):
            r = f.read(4 + 1 + 8 * 9 + 1)
            s = struct.unpack('>ic9dB', r)
            if date_range[0] <= round(s[0]) <= date_range[1]:
                ACCd = np.zeros(4)
                '''time'''
                ACCd[0] = s[0]
                '''X'''
                ACCd[1] = s[2]
                '''Y'''
                ACCd[2] = s[3]
                '''Z'''
                ACCd[3] = s[4]
                ACC.append(ACCd)
        f.close()

        return ACC

    def _readSCA(self, date: str, sat: SatID):
        path = str(self.dir.joinpath(self._path_SCA, 'grace_1B_' + date.split('-')[0] + '-' + date.split('-')[1] + '_03',
                              'SCA1B_' + date + '_' + sat.name + '_03.dat'))
        st = date.split('-')
        IY, IM, ID = int(st[0]), int(st[1]), int(st[2])
        date_range = (round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 0, 0, 0))),
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.SCA.name]))))
        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
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

        # SCA = np.zeros((NumOfSCA, 5))

        SCA = []
        for i in range(NumOfSCA):
            r = f.read(4 + 1 + 1 + 8 * 5 + 1)
            if len(r) != 47:
                continue
            else:
                s = struct.unpack('>icb5dB', r)
                if date_range[0] <= round(s[0]) <= date_range[1]:
                    SCAd = np.zeros(5)
                    '''time'''
                    SCAd[0] = s[0]
                    '''cos(mu/2)'''
                    SCAd[1] = s[3]
                    '''I'''
                    SCAd[2] = s[4]
                    '''J'''
                    SCAd[3] = s[5]
                    '''K'''
                    SCAd[4] = s[6]
                    SCA.append(SCAd)
                else:
                    continue
        f.close()

        return SCA

    def _readKBR(self, date: str):
        path = str(self.dir.joinpath(self._path_KBR, 'grace_1B_' + date.split('-')[0] + '-' + date.split('-')[1] + '_03',
                              'KBR1B_' + date + '_X_03.dat'))

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
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
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.GNV.name]))))

        path = str(self.dir.joinpath(self._path_GNV, date.split('-')[0] + '-' + date.split('-')[1],
                              'grace_1B_' + date + '_02', 'GNV1B_' + date + '_' + sat.name + '_02.dat'))

        try:
            f = open(path, 'rb')
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
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
        # assert self._InterfaceConfig.mission == Mission.GRACE_FO_RL04.name
        pass

    def _readACC(self, date: str, sat: SatID):
        path = str(self.dir.joinpath(self._path_ACC, date.split('-')[0] + '-' + date.split('-')[1],
                                     'gracefo_1B_' + date + '_RL04.ascii.noLRI.tgz_files',
                                     'ACT1B_' + date + '_' + chr(ord(sat.name) + 2) + '_04.txt'))
        try:
            f = open(path, 'r')
        except FileNotFoundError as e:
            # print(path)
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

    # def _readSCA(self, date: str, sat: SatID):
    #     path = str(self.dir.joinpath(self._path_SCA, date.split('-')[0] + '-' + date.split('-')[1],
    #                                  'gracefo_1B_' + date + '_RL04.ascii.noLRI.tgz_files',
    #                                  'SCA1B_' + date + '_' + chr(ord(sat.name) + 2) + '_04.txt'))
    #     try:
    #         f = open(path, 'r')
    #     except FileNotFoundError as e:
    #         # print(path)
    #         return None
    #
    #     NumOfSCA = None
    #     while True:
    #         r = f.readline()
    #         if r.split(':')[0].strip() == 'num_records':
    #             NumOfSCA = int(r.split(':')[1])
    #         if r.strip() == '# End of YAML header':
    #             break
    #
    #     # SCA = np.zeros((NumOfSCA, 5))
    #     # SCA = []
    #     # for i in range(NumOfSCA):
    #     #     s = f.readline().split()
    #     #     if len(s) == 9:
    #     #         SCAd = np.zeros(5)
    #     #         '''time'''
    #     #         SCAd[0] = s[0]
    #     #         '''cos(mu/2)'''
    #     #         SCAd[1] = s[3]
    #     #         '''I'''
    #     #         SCAd[2] = s[4]
    #     #         '''J'''
    #     #         SCAd[3] = s[5]
    #     #         '''K'''
    #     #         SCAd[4] = s[6]
    #     #         SCA.append(SCAd)
    #     #     else:
    #     #         continue
    #
    #     SCA = np.zeros((NumOfSCA, 5))
    #
    #     for i in range(NumOfSCA):
    #         s = f.readline().split()
    #         '''time'''
    #         SCA[i, 0] = float(s[0])
    #         '''cos(mu/2)'''
    #         SCA[i, 1] = s[3]
    #         '''I'''
    #         SCA[i, 2] = s[4]
    #         '''J'''
    #         SCA[i, 3] = s[5]
    #         '''K'''
    #         SCA[i, 4] = s[6]
    #     f.close()
    #
    #     return SCA

    def _readKBR(self, date: str):
        path = str(self.dir.joinpath(self._path_KBR, date.split('-')[0] + '-' + date.split('-')[1],
                                     'gracefo_1B_' + date + '_RL04.ascii.noLRI.tgz_files',
                                     'KBR1B_' + date + '_Y_04.txt'))

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
        if NumOfKBR == 0:
            return np.zeros((NumOfKBR, 4))
        # KBR = np.zeros((NumOfKBR, 4))
        KBR = []
        for i in range(NumOfKBR):
            s = f.readline().split()
            if len(s) == 16:
                KBRd = np.zeros(4)
                '''time'''
                KBRd[0] = s[0]
                '''bias range + light_corr + ant_corr '''
                KBRd[1] = float(s[1]) + float(s[5]) + float(s[8])
                '''bias range-rate + light_corr + ant_corr '''
                KBRd[2] = float(s[2]) + float(s[6]) + float(s[9])
                '''bias range-acc + light_corr + ant_corr '''
                KBRd[3] = float(s[3]) + float(s[7]) + float(s[10])
                KBR.append(KBRd)
            else:
                continue
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
                      round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.sr_payload[Payload.GNV.name]))))
        path = str(self.dir.joinpath(self._path_GNV, date.split('-')[0] + '-' + date.split('-')[1],
                                     'gracefo_1B_' + date + '_RL04.ascii.noLRI.tgz_files',
                                     'GNV1B_' + date + '_' + chr(ord(sat.name) + 2) + '_04.txt'))
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
        if NumOfGNV == 0:
            return np.array(np.zeros((NumOfGNV, 7)))

        GNV = []

        for i in range(NumOfGNV):
            """delete the data out of the range of given date"""
            s = f.readline().split()
            if len(s) == 16:
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
            else:
                continue

        f.close()
        return np.array(GNV)

    # def _readACC(self, date: str, sat: SatID):
    #     path = str(self.dir.joinpath(self._path_ACC, 'ACC-' + date.split('-')[0] + '-' + date.split('-')[1],
    #                                  'ACT1B_' + date + '-' + chr(ord(sat.name) + 2) + '-01.bin'))
    #
    #     data = np.fromfile(path, dtype=np.float64)
    #
    #     ACT = np.array(data).reshape((-1, 4))
    #
    #     return ACT

    def _readSCA(self, date: str, sat: SatID):
        path = str(self.dir.joinpath(self._path_SCA, date.split('-')[0] + '-' + date.split('-')[1],
                                     date,
                                     'SCA1B_' + date + '_' + chr(ord(sat.name) + 2) + '_04.txt'))
        try:
            f = open(path, 'r')
        except FileNotFoundError as e:
            # print(path)
            return None

        NumOfSCA = None
        while True:
            r = f.readline()
            if r.split(':')[0].strip() == 'num_records':
                NumOfSCA = int(r.split(':')[1])
            if r.strip() == '# End of YAML header':
                break

        # SCA = np.zeros((NumOfSCA, 5))
        # SCA = []
        # for i in range(NumOfSCA):
        #     s = f.readline().split()
        #     if len(s) == 9:
        #         SCAd = np.zeros(5)
        #         '''time'''
        #         SCAd[0] = s[0]
        #         '''cos(mu/2)'''
        #         SCAd[1] = s[3]
        #         '''I'''
        #         SCAd[2] = s[4]
        #         '''J'''
        #         SCAd[3] = s[5]
        #         '''K'''
        #         SCAd[4] = s[6]
        #         SCA.append(SCAd)
        #     else:
        #         continue

        SCA = np.zeros((NumOfSCA, 5))

        for i in range(NumOfSCA):
            s = f.readline().split()
            '''time'''
            SCA[i, 0] = float(s[0])
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
