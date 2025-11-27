import numpy as np
from tqdm import trange
import os
from tqdm import tqdm
from src.Frame.Frame import Frame
from src.Preference.EnumType import SatID, Payload
from src.Interface.L1b import L1b as Level_1B
import pathlib


class KinematicOrbit:
    """
    The class deals with the loading and the preprocessing of the kinematic orbit, where only ITSG option is offered.
    """
    def __init__(self, L1b: Level_1B):
        self.dir = pathlib.Path()
        self._L1b = L1b
        self._KiOr_A = None
        self._KiOr_B = None
        pass

    def configure(self, **kwargs):
        self._path_kinematic = self._L1b.InterfacePathConfig.payload_Kinematic

        '''isGCRS: tell the system that what's the frame of the GNV orbit. True--GCRS, False--ITRS'''
        self.isGCRS = self._L1b.InterfaceConfig.is_orbit_GCRS[Payload.KinematicOrbit.name]

        return self

    def _read_by_day(self, date: str, sat: SatID):
        pass

    def read_single_sat(self, sat: SatID):
        KiOrbit = np.zeros((0, 4))

        datelist = self._L1b.getDate()
        for i in trange(len(datelist), desc='Kinematic orbit reading for %s' % sat,
                        disable=os.environ.get("DISABLE_TQDM", False)):
            res = self._read_by_day(sat=sat, date=datelist[i])
            if res is not None:
                KiOrbit = np.vstack((KiOrbit, res))
            pass

        return KiOrbit

    def read_double_sat(self):
        self._KiOr_A = self.read_single_sat(SatID.A)
        self._KiOr_B = self.read_single_sat(SatID.B)

        '''pass to the L1b dataset'''
        self._L1b.KiOrb_A = self._KiOr_A
        self._L1b.KiOrb_B = self._KiOr_B
        pass

    def getKiOrbit(self, sat: SatID):
        if sat == SatID.A:
            return self._KiOr_A
        else:
            return self._KiOr_B

    def GNVandKiOrbitItrs2Gcrs(self, fr: Frame):
        """
        Convert the GNV orbit and Kinematic orbit from ITRS frame into GCRS frame
        :return:
        """
        if self._KiOr_A is None:
            '''do nothing if no kinematic orbit data'''
            self.isGCRS = True

        if self._L1b.is_GNV_GCRS and self.isGCRS:
            '''do nothing'''
            pass
        elif self._L1b.is_GNV_GCRS and (not self.isGCRS):
            '''convert kinematic orbit'''
            self.__CoordinateTrans3(fr)

        elif (not self._L1b.is_GNV_GCRS) and self.isGCRS:
            '''convert GNV orbit'''
            self.__CoordinateTrans2(fr)
        else:
            '''convert them simultaneously'''
            self.__CoordinateTrans1(fr)
            pass

        pass

    def __CoordinateTrans1(self, fr: Frame):
        GNVA = self._L1b.getData(Payload.GNV.name, SatID.A)
        GNVB = self._L1b.getData(Payload.GNV.name, SatID.B)
        KiA = self._KiOr_A
        KiB = self._KiOr_B

        timeGNVA, timeGNVB, timeKiA, timeKiB = GNVA[:, 0].astype(np.int64), GNVB[:, 0].astype(np.int64), \
                                               KiA[:, 0].astype(np.int64), KiB[:, 0].astype(np.int64)

        commonT = set(timeGNVA).union(set(timeGNVB)).union(set(timeKiA)).union(set(timeKiB))
        commonT = list(commonT)
        commonT.sort()

        icrs_GNVA_pos = []
        icrs_GNVA_vel = []
        icrs_GNVB_pos = []
        icrs_GNVB_vel = []

        icrs_KiA_pos = []
        icrs_KiB_pos = []

        n, m, j, k = 0, 0, 0, 0
        for time in tqdm(commonT, desc='Orbit conversion from ITRS into GCRS: '):
            rm = fr.setTime(time).getRotationMatrix

            if time <= timeGNVA[-1] and time == timeGNVA[n]:
                icrs_GNVA_pos.append(Frame.PosITRS2GCRS(GNVA[n, 1:4], rm))
                icrs_GNVA_vel.append(Frame.VelITRS2GCRS(GNVA[n, 1:4], GNVA[n, 4:7], rm))
                n += 1
                pass

            if time <= timeGNVB[-1] and time == timeGNVB[m]:
                icrs_GNVB_pos.append(Frame.PosITRS2GCRS(GNVB[m, 1:4], rm))
                icrs_GNVB_vel.append(Frame.VelITRS2GCRS(GNVB[m, 1:4], GNVB[m, 4:7], rm))
                m += 1
                pass

            if time <= timeKiA[-1] and time == timeKiA[j]:
                icrs_KiA_pos.append(Frame.PosITRS2GCRS(KiA[j, 1:4], rm))
                j += 1
                pass

            if time <= timeKiB[-1] and time == timeKiB[k]:
                icrs_KiB_pos.append(Frame.PosITRS2GCRS(KiB[k, 1:4], rm))
                k += 1
                pass

            pass

        assert n == len(timeGNVA) and m == len(timeGNVB) \
               and j == len(timeKiA) and k == len(timeKiB), 'Error when transform orbit from ITRS into GCRS!!'

        x = np.hstack((np.array(icrs_GNVA_pos), np.array(icrs_GNVA_vel)))
        self._L1b.GNV_A = np.hstack((GNVA[:, 0][:, None], x))

        x = np.hstack((np.array(icrs_GNVB_pos), np.array(icrs_GNVB_vel)))
        self._L1b.GNV_B = np.hstack((GNVB[:, 0][:, None], x))

        self._KiOr_A = np.hstack((KiA[:, 0][:, None], np.array(icrs_KiA_pos)))
        self._L1b.KiOrb_A = self._KiOr_A

        self._KiOr_B = np.hstack((KiB[:, 0][:, None], np.array(icrs_KiB_pos)))
        self._L1b.KiOrb_B = self._KiOr_B

        pass

    def __CoordinateTrans2(self, fr: Frame):
        GNVA = self._L1b.getData(Payload.GNV.name, SatID.A)
        GNVB = self._L1b.getData(Payload.GNV.name, SatID.B)

        timeGNVA, timeGNVB = GNVA[:, 0].astype(np.int64), GNVB[:, 0].astype(np.int64)

        commonT = set(timeGNVA).union(set(timeGNVB))
        commonT = list(commonT)
        commonT.sort()

        icrs_GNVA_pos = []
        icrs_GNVA_vel = []
        icrs_GNVB_pos = []
        icrs_GNVB_vel = []

        n, m = 0, 0
        for time in tqdm(commonT, desc='Orbit conversion from ITRS into GCRS: '):
            rm = fr.setTime(time).getRotationMatrix
            if time <= timeGNVA[-1] and time == timeGNVA[n]:
                icrs_GNVA_pos.append(Frame.PosITRS2GCRS(GNVA[n, 1:4], rm))
                icrs_GNVA_vel.append(Frame.VelITRS2GCRS(GNVA[n, 1:4], GNVA[n, 4:7], rm))
                n += 1
                pass

            if time <= timeGNVB[-1] and time == timeGNVB[m]:
                icrs_GNVB_pos.append(Frame.PosITRS2GCRS(GNVB[m, 1:4], rm))
                icrs_GNVB_vel.append(Frame.VelITRS2GCRS(GNVB[m, 1:4], GNVB[m, 4:7], rm))
                m += 1
                pass

            pass

        assert n == len(timeGNVA) and m == len(timeGNVB), 'Error when transform orbit from ITRS into GCRS!!'

        x = np.hstack((np.array(icrs_GNVA_pos), np.array(icrs_GNVA_vel)))
        self._L1b.GNV_A = np.hstack((GNVA[:, 0][:, None], x))

        x = np.hstack((np.array(icrs_GNVB_pos), np.array(icrs_GNVB_vel)))
        self._L1b.GNV_B = np.hstack((GNVB[:, 0][:, None], x))

        pass

    def __CoordinateTrans3(self, fr: Frame):
        KiA = self._KiOr_A
        KiB = self._KiOr_B

        timeKiA, timeKiB = KiA[:, 0].astype(np.int64), KiB[:, 0].astype(np.int64)

        commonT = set(timeKiA).union(set(timeKiB))
        commonT = list(commonT)
        commonT.sort()

        icrs_KiA_pos = []
        icrs_KiB_pos = []

        j, k = 0, 0
        for time in tqdm(commonT, desc='Orbit conversion from ITRS into GCRS: '):
            rm = fr.setTime(time).getRotationMatrix

            if time <= timeKiA[-1] and time == timeKiA[j]:
                icrs_KiA_pos.append(Frame.PosITRS2GCRS(KiA[j, 1:4], rm))
                j += 1
                pass

            if time <= timeKiB[-1] and time == timeKiB[k]:
                icrs_KiB_pos.append(Frame.PosITRS2GCRS(KiB[k, 1:4], rm))
                k += 1
                pass

            pass

        assert j == len(timeKiA) and k == len(timeKiB), 'Error when transform orbit from ITRS into GCRS!!'

        self._KiOr_A = np.hstack((KiA[:, 0][:, None], np.array(icrs_KiA_pos)))
        self._L1b.KiOrb_A = self._KiOr_A

        self._KiOr_B = np.hstack((KiB[:, 0][:, None], np.array(icrs_KiB_pos)))
        self._L1b.KiOrb_B = self._KiOr_B

        pass


class KinematicOrbitV3(KinematicOrbit):
    """
    The class deals with the loading and the preprocessing of the kinematic orbit, where only ITSG option is offered.
    """
    def __init__(self, L1b: Level_1B):
        super(KinematicOrbit, self).__init__()
        self.__L1b = L1b

        self.__KiOr_A = None
        self.__KiOr_B = None
        pass

    def configure(self, **kwargs):
        self.__path_kinematic = self.__L1b.InterfacePathConfig.payload_Kinematic

        '''isGCRS: tell the system that what's the frame of the GNV orbit. True--GCRS, False--ITRS'''
        self.isGCRS = self.__L1b.InterfaceConfig.is_orbit_GCRS[Payload.KinematicOrbit.name]

        return self

    def _read_by_day(self, date: str, sat: SatID):
        """
        Input is in ITRS frame, which has to be transferred to GCRS inertial frame
        :param date:
        :param sat:
        :return:
        """

        st = date.split('-')
        IY, IM, ID = int(st[0]), int(st[1]), int(st[2])
        # date_range = (round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 0, 0, 0))),
        #               round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self.__L1b.sr_payload[Payload.GNV.name]))))

        path = self.__path_kinematic.joinpath(*[str(IY), date.split('-')[0] + '-' + date.split('-')[1], 'grace' +
                                                str(sat.value) + '-kinematicOrbit-' + date + '.txt'])

        res = None

        try:
            res = np.loadtxt(path, dtype=np.float64, comments='#', usecols=(0, 1, 2, 3))
            res[:, 0] = np.round(Frame.mjd2sec(res[:, 0]))
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
            return None

        return res

    def read_single_sat(self, sat: SatID):
        KiOrbit = np.zeros((0, 4))

        datelist = self.__L1b.getDate()
        for i in trange(len(datelist), desc='Kinematic orbit reading for %s' % sat,
                        disable=os.environ.get("DISABLE_TQDM", False)):
            res = self._read_by_day(sat=sat, date=datelist[i])
            if res is not None:
                KiOrbit = np.vstack((KiOrbit, res))
            pass

        return KiOrbit

    def read_double_sat(self):
        self.__KiOr_A = self.read_single_sat(SatID.A)
        self.__KiOr_B = self.read_single_sat(SatID.B)

        '''pass to the L1b dataset'''
        self.__L1b.KiOrb_A = self.__KiOr_A
        self.__L1b.KiOrb_B = self.__KiOr_B
        pass

    def getKiOrbit(self, sat: SatID):
        if sat == SatID.A:
            return self.__KiOr_A
        else:
            return self.__KiOr_B

    def GNVandKiOrbitItrs2Gcrs(self, fr: Frame):
        """
        Convert the GNV orbit and Kinematic orbit from ITRS frame into GCRS frame
        :return:
        """
        if self.__KiOr_A is None:
            '''do nothing if no kinematic orbit data'''
            self.isGCRS = True

        if self.__L1b.is_GNV_GCRS and self.isGCRS:
            '''do nothing'''
            pass
        elif self.__L1b.is_GNV_GCRS and (not self.isGCRS):
            '''convert kinematic orbit'''
            self.__CoordinateTrans3(fr)

        elif (not self.__L1b.is_GNV_GCRS) and self.isGCRS:
            '''convert GNV orbit'''
            self.__CoordinateTrans2(fr)
        else:
            '''convert them simultaneously'''
            self.__CoordinateTrans1(fr)
            pass

        pass

    def __CoordinateTrans1(self, fr: Frame):
        GNVA = self.__L1b.getData(Payload.GNV.name, SatID.A)
        GNVB = self.__L1b.getData(Payload.GNV.name, SatID.B)
        KiA = self.__KiOr_A
        KiB = self.__KiOr_B

        timeGNVA, timeGNVB, timeKiA, timeKiB = GNVA[:, 0].astype(np.int64), GNVB[:, 0].astype(np.int64), \
                                               KiA[:, 0].astype(np.int64), KiB[:, 0].astype(np.int64)

        commonT = set(timeGNVA).union(set(timeGNVB)).union(set(timeKiA)).union(set(timeKiB))
        commonT = list(commonT)
        commonT.sort()

        icrs_GNVA_pos = []
        icrs_GNVA_vel = []
        icrs_GNVB_pos = []
        icrs_GNVB_vel = []

        icrs_KiA_pos = []
        icrs_KiB_pos = []

        n, m, j, k = 0, 0, 0, 0
        for time in tqdm(commonT, desc='Orbit conversion from ITRS into GCRS: '):
            rm = fr.setTime(time).getRotationMatrix
            if time <= timeGNVA[-1] and time == timeGNVA[n]:
                icrs_GNVA_pos.append(Frame.PosITRS2GCRS(GNVA[n, 1:4], rm))
                icrs_GNVA_vel.append(Frame.VelITRS2GCRS(GNVA[n, 1:4], GNVA[n, 4:7], rm))
                n += 1
                pass

            if time <= timeGNVB[-1] and time == timeGNVB[m]:
                icrs_GNVB_pos.append(Frame.PosITRS2GCRS(GNVB[m, 1:4], rm))
                icrs_GNVB_vel.append(Frame.VelITRS2GCRS(GNVB[m, 1:4], GNVB[m, 4:7], rm))
                m += 1
                pass

            if time <= timeKiA[-1] and time == timeKiA[j]:
                icrs_KiA_pos.append(Frame.PosITRS2GCRS(KiA[j, 1:4], rm))
                j += 1
                pass

            if time <= timeKiB[-1] and time == timeKiB[k]:
                icrs_KiB_pos.append(Frame.PosITRS2GCRS(KiB[k, 1:4], rm))
                k += 1
                pass

            pass

        assert n == len(timeGNVA) and m == len(timeGNVB) \
               and j == len(timeKiA) and k == len(timeKiB), 'Error when transform orbit from ITRS into GCRS!!'

        x = np.hstack((np.array(icrs_GNVA_pos), np.array(icrs_GNVA_vel)))
        self.__L1b.GNV_A = np.hstack((GNVA[:, 0][:, None], x))

        x = np.hstack((np.array(icrs_GNVB_pos), np.array(icrs_GNVB_vel)))
        self.__L1b.GNV_B = np.hstack((GNVB[:, 0][:, None], x))

        self.__KiOr_A = np.hstack((KiA[:, 0][:, None], np.array(icrs_KiA_pos)))
        self.__L1b.KiOrb_A = self.__KiOr_A

        self.__KiOr_B = np.hstack((KiB[:, 0][:, None], np.array(icrs_KiB_pos)))
        self.__L1b.KiOrb_B = self.__KiOr_B

        pass

    def __CoordinateTrans2(self, fr: Frame):
        GNVA = self.__L1b.getData(Payload.GNV.name, SatID.A)
        GNVB = self.__L1b.getData(Payload.GNV.name, SatID.B)

        timeGNVA, timeGNVB = GNVA[:, 0].astype(np.int64), GNVB[:, 0].astype(np.int64)

        commonT = set(timeGNVA).union(set(timeGNVB))
        commonT = list(commonT)
        commonT.sort()

        icrs_GNVA_pos = []
        icrs_GNVA_vel = []
        icrs_GNVB_pos = []
        icrs_GNVB_vel = []

        n, m = 0, 0
        for time in tqdm(commonT, desc='Orbit conversion from ITRS into GCRS: '):
            rm = fr.setTime(time).getRotationMatrix
            if time <= timeGNVA[-1] and time == timeGNVA[n]:
                icrs_GNVA_pos.append(Frame.PosITRS2GCRS(GNVA[n, 1:4], rm))
                icrs_GNVA_vel.append(Frame.VelITRS2GCRS(GNVA[n, 1:4], GNVA[n, 4:7], rm))
                n += 1
                pass

            if time <= timeGNVB[-1] and time == timeGNVB[m]:
                icrs_GNVB_pos.append(Frame.PosITRS2GCRS(GNVB[m, 1:4], rm))
                icrs_GNVB_vel.append(Frame.VelITRS2GCRS(GNVB[m, 1:4], GNVB[m, 4:7], rm))
                m += 1
                pass

            pass

        assert n == len(timeGNVA) and m == len(timeGNVB), 'Error when transform orbit from ITRS into GCRS!!'

        x = np.hstack((np.array(icrs_GNVA_pos), np.array(icrs_GNVA_vel)))
        self.__L1b.GNV_A = np.hstack((GNVA[:, 0][:, None], x))

        x = np.hstack((np.array(icrs_GNVB_pos), np.array(icrs_GNVB_vel)))
        self.__L1b.GNV_B = np.hstack((GNVB[:, 0][:, None], x))

        pass

    def __CoordinateTrans3(self, fr: Frame):
        KiA = self.__KiOr_A
        KiB = self.__KiOr_B

        timeKiA, timeKiB = KiA[:, 0].astype(np.int64), KiB[:, 0].astype(np.int64)

        commonT = set(timeKiA).union(set(timeKiB))
        commonT = list(commonT)
        commonT.sort()

        icrs_KiA_pos = []
        icrs_KiB_pos = []

        j, k = 0, 0
        for time in tqdm(commonT, desc='Orbit conversion from ITRS into GCRS: '):
            rm = fr.setTime(time).getRotationMatrix

            if time <= timeKiA[-1] and time == timeKiA[j]:
                icrs_KiA_pos.append(Frame.PosITRS2GCRS(KiA[j, 1:4], rm))
                j += 1
                pass

            if time <= timeKiB[-1] and time == timeKiB[k]:
                icrs_KiB_pos.append(Frame.PosITRS2GCRS(KiB[k, 1:4], rm))
                k += 1
                pass

            pass

        assert j == len(timeKiA) and k == len(timeKiB), 'Error when transform orbit from ITRS into GCRS!!'

        self.__KiOr_A = np.hstack((KiA[:, 0][:, None], np.array(icrs_KiA_pos)))
        self.__L1b.KiOrb_A = self.__KiOr_A

        self.__KiOr_B = np.hstack((KiB[:, 0][:, None], np.array(icrs_KiB_pos)))
        self.__L1b.KiOrb_B = self.__KiOr_B

        pass


class KinematicOrbitV2(KinematicOrbit):
    """
    The class deals with the loading and the preprocessing of the kinematic orbit, where only ITSG option is offered.
    """
    def __init__(self, L1b: Level_1B):
        super(KinematicOrbitV2, self).__init__(L1b)
        pass

    def _read_by_day(self, date: str, sat: SatID):
        """
        Input is in ITRS frame, which has to be transferred to GCRS inertial frame
        :param date:
        :param sat:
        :return:
        """

        st = date.split('-')
        IY, IM, ID = int(st[0]), int(st[1]), int(st[2])
        # date_range = (round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 0, 0, 0))),
        #               round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self._L1b.sr_payload[Payload.GNV.name]))))

        path = str(self.dir.joinpath(*[self._path_kinematic, str(IY), date.split('-')[0] + '-' + date.split('-')[1], 'Grace' +
                                                str(sat.name) + '-kinematicOrbit-' + date, 'grace' +
                                                str(sat.name) + '-kinOrb-' + date + '.txt']))
        res = None
        if not os.path.exists(path):
            return None
        try:
            res = np.loadtxt(path, dtype=np.float64, comments='#', usecols=(0, 1, 2, 3))
            if np.shape(res)[0] == 0:
                return None
            res[:, 0] = np.round(Frame.mjd2sec(res[:, 0]))
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
            return None

        return res


class KinematicOrbitFO(KinematicOrbit):
    """
    The class deals with the loading and the preprocessing of the kinematic orbit, where only ITSG option is offered.
    """
    def __init__(self, L1b: Level_1B):
        super(KinematicOrbitFO, self).__init__(L1b)
        pass

    def _read_by_day(self, date: str, sat: SatID):
        """
        Input is in ITRS frame, which has to be transferred to GCRS inertial frame
        :param date:
        :param sat:
        :return:
        """

        st = date.split('-')
        IY, IM, ID = int(st[0]), int(st[1]), int(st[2])
        # date_range = (round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 0, 0, 0))),
        #               round(Frame.mjd2sec(Frame.cal2mjd(IY, IM, ID, 23, 59, 60 - self._L1b.sr_payload[Payload.GNV.name]))))

        path = str(self.dir.joinpath(*[self._path_kinematic, str(IY), date.split('-')[0] + '-' + date.split('-')[1],
                   'GRACEFO-' + str(sat.value) + '_kinematicOrbit_' + date + '.txt']))
        res = None
        if not os.path.exists(path):
            return None
        try:
            # res = np.loadtxt(path, dtype=np.float64, comments='#', usecols=(0, 1, 2, 3))
            res = np.genfromtxt(path, delimiter=None, dtype=np.float64, skip_header=6, usecols=(0, 1, 2, 3))
            if np.shape(res)[0] == 0:
                return None
            res[:, 0] = np.round(Frame.mjd2sec(res[:, 0]))
        except FileNotFoundError as e:
            # lg.error(e)
            # print(path)
            return None

        return res

