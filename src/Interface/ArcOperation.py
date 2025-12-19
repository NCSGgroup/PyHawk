import numpy as np
import datetime
import h5py
from tqdm import tqdm
from src.Preference.EnumType import SatID, Payload
from src.Interface.GapFix import GapFix
from src.Preference.Pre_Interface import InterfaceConfig
from src.Auxilary.Outlier import Outlier
import pathlib
import os


class ArcSelect:
    """
    get the raw data to be computed and divide them arc by arc. Every single arc has no discontinuity
    """
    def __init__(self, gf: GapFix):
        self.__duration = None
        self.__arcs = None
        self.__gf = gf
        self.__h5py = None

    def configure(self, **kwargs):
        self.__arc_length = self.__gf.L1b.InterfaceConfig.arc_length
        self.__min_arcLen = self.__gf.L1b.InterfaceConfig.min_arcLen
        self.__arc_path = self.__gf.L1b.InterfacePathConfig.report_arc
        self.__arcft_path = self.__gf.L1b.InterfacePathConfig.report_arcft
        self.__raw_data_path = self.__gf.L1b.InterfacePathConfig.temp_raw_data
        self.__sr_target = self.__gf.L1b.InterfaceConfig.sr_target
        self.__OrbitConfig = self.__gf.L1b.OrbitConfig
        return self

    def unpackArcs(self):

        # ------------------------loading data----------------------------------
        ACC_A = self.__gf.getData(Payload.ACC.name, SatID.A)[0]
        ACC_B = self.__gf.getData(Payload.ACC.name, SatID.B)[0]
        SCA_A = self.__gf.getData(Payload.SCA.name, SatID.A)[0]
        SCA_B = self.__gf.getData(Payload.SCA.name, SatID.B)[0]
        GNV_A = self.__gf.getData(Payload.GNV.name, SatID.A)[0]
        GNV_B = self.__gf.getData(Payload.GNV.name, SatID.B)[0]
        KBR = self.__gf.getData(Payload.KBR.name, SatID.A)[0]
        KiOrbA = self.__gf.L1b.getData(Payload.KinematicOrbit.name, SatID.A)
        KiOrbB = self.__gf.L1b.getData(Payload.KinematicOrbit.name, SatID.B)
        # ------------------------outlier kinOrbit data----------------------------------
        # KiOrbA = self.__outlier(GNV_A, KiOrbA)
        # KiOrbB = self.__outlier(GNV_B, KiOrbB)
        # -------------------------arc divide--------------------------------------
        timelist = [ACC_A[:, 0].copy(), ACC_B[:, 0].copy(), SCA_A[:, 0].copy(), SCA_B[:, 0].copy()]

        self.__divide2(timelist)
        self.__discardArcs()

        date = self.__gf.getDate()
        filename = self.__raw_data_path + '/' + date[0] + '_' + date[-1] + '.hdf5'
        # self.__h5py = h5py.File(self.data_path + date[0] + '_' + date[-1] + ".hdf5", "w")
        self.__h5py = h5py.File(filename, "w")
        # d1 = f.create_dataset("ACC_A", data=ACC_A)
        if Payload.ACC.name in self.__gf.L1b.include_payloads:
            self.__unpack(ACC_A, Payload.ACC, SatID.A)
            self.__unpack(ACC_B, Payload.ACC, SatID.B)
        if Payload.SCA.name in self.__gf.L1b.include_payloads:
            self.__unpack(SCA_A, Payload.SCA, SatID.A)
            self.__unpack(SCA_B, Payload.SCA, SatID.B)
        if Payload.GNV.name in self.__gf.L1b.include_payloads:
            self.__unpack(GNV_A, Payload.GNV, SatID.A)
            self.__unpack(GNV_B, Payload.GNV, SatID.B)
        if Payload.KBR.name in self.__gf.L1b.include_payloads:
            self.__unpack(KBR, Payload.KBR, SatID.A)
        if Payload.KinematicOrbit.name in self.__gf.L1b.include_payloads:
            self.__unpack(data=KiOrbA, sat=SatID.A, Gr=Payload.KinematicOrbit)
            self.__unpack(data=KiOrbB, sat=SatID.B, Gr=Payload.KinematicOrbit)

        self.__h5py.close()
        if Payload.LRI.name in self.__gf.L1b.include_payloads:
            pass

        return self

    def __divide1(self, timelist):
        """
        All is given in GPS time
        :param timelist: 0: ACC-A; 1: ACC-B, 2:SCA-A, 3:SCA-B
        :return:
        """
        stepsize = self.__sr_target
        # stepsize = 5

        '''find the intersection of four instruments: ACC-A, ACC-B, SCA-A, SCA-B'''
        commonTime = set(timelist[0].astype(np.int64))
        for i in range(1, len(timelist)):
            commonTime = commonTime & set(timelist[i].astype(np.int64))

        commonTime = np.array(list(commonTime))
        commonTime.sort()

        '''hour to sec, then to the index'''
        al = int(self.__arc_length * 3600 / stepsize)
        amin = int(self.__min_arcLen * 3600 / stepsize)
        begin = commonTime[0]
        end = commonTime[-1]

        '''create a virtual time arr that has no discontinuity but marks the commonTime '''
        VirtualTime = np.arange(begin, end + 1, stepsize)
        indexOfCommon = np.zeros(len(VirtualTime), dtype=bool)
        indexOfCommon[((commonTime - commonTime[0]) / stepsize).astype(np.int64)] = True

        '''basic arc division defined by arc length'''
        arcD1 = np.arange(0, len(VirtualTime), al)
        arclist0 = []
        for i in range(len(arcD1) - 1):
            arclist0.append(np.arange(arcD1[i], arcD1[i + 1]))
        arclist0.append(np.arange(arcD1[-1], len(VirtualTime)))

        '''arc division considering the gaps'''
        arcs = []
        for term in arclist0:
            li = ArcSelect.divideArcByBool(indexOfCommon[term])

            if li is None:
                arcs.append(VirtualTime[term])
                continue

            for lii in li:
                arcs.append(VirtualTime[term[lii]])

        self.__arcs = arcs

        pass

    def __divide2(self, timelist):
        """
        All is given in GPS time
        :param timelist: 0: ACC-A; 1: ACC-B, 2:SCA-A, 3:SCA-B
        :return:
        """

        # stepsize = 5

        stepsize = self.__sr_target
        '''find the intersection of four instruments: ACC-A, ACC-B, SCA-A, SCA-B'''
        # commonTime = set(timelist[0].astype(np.int64)) & set(timelist[1].astype(np.int64)) \
        #              & set(timelist[2].astype(np.int64)) & set(timelist[3].astype(np.int64))
        commonTime = set(timelist[0].astype(np.int64))
        for i in range(1, len(timelist)):
            commonTime = commonTime & set(timelist[i].astype(np.int64))

        commonTime = np.array(list(commonTime))
        commonTime.sort()
        commonTime = commonTime[commonTime % stepsize == 0]

        '''hour to sec, then to the index'''
        al = int(self.__arc_length * 3600 / stepsize)
        amin = int(self.__min_arcLen * 3600 / stepsize)
        begin = commonTime[0]
        end = commonTime[-1]

        '''create a virtual time arr that has no discontinuity but marks the commonTime '''
        VirtualTime = np.arange(begin, end + 1, stepsize)
        indexOfCommon = np.zeros(len(VirtualTime), dtype=bool)

        indexOfCommon[((commonTime - commonTime[0]) / stepsize).astype(np.int64)] = True

        '''Gaps will first divide the data into several arcs'''
        li = ArcSelect.divideArcByBool(indexOfCommon)

        '''For each arc, the given arc length will further divide them into smaller arcs'''
        arcs = []
        for term in li:
            if len(term) <= al:
                arcs.append(VirtualTime[term])
                continue
            arcD1 = np.arange(term[0], term[-1], al)
            for i in range(len(arcD1) - 1):
                arcs.append(VirtualTime[np.arange(arcD1[i], arcD1[i + 1])])
            arcs.append(VirtualTime[np.arange(arcD1[-1], term[-1] + 1)])

        self.__arcs = arcs

        pass

    def __discardArcs(self):
        # stepsize = 5
        stepsize = self.__sr_target
        amin = int(self.__min_arcLen * 3600 / stepsize)

        def delarc(elem):
            return (elem[-1] - elem[0]) / stepsize + 1 >= amin

        self.__arcs = list(filter(delarc, self.__arcs))
        pass

    def __unpack(self, data, Gr: Payload, sat: SatID):
        """
        Automatically unpack the data into several arcs.
        :param data: one instrument data that waits to be divided.
        :return:
        """

        arcs = self.__arcs
        name = Gr.name + '_' + sat.name

        rmsx = self.rms(data[:, 1])
        for i, arc in enumerate(tqdm(arcs, desc='Arc unpacking for ' + name), 0):
            time = data[:, 0].astype(np.int64)
            x1 = data[time >= arc[0], :]
            time = x1[:, 0].astype(np.int64)
            x1 = x1[time <= arc[-1], :]
            time = x1[:, 0].astype(np.int64)
            '''resample to make data consistent'''
            x1 = x1[time % self.__sr_target == 0, :]
            self.__h5py.create_dataset(name + '_arc' + str(i), data=x1)

        pass

    def getArcs(self):
        return self.__arcs

    def makeReport(self):

        date = self.__gf.getDate()
        path = self.__arc_path + '/' + date[0] + '_' + date[-1] + '.txt'
        with open(path, 'w') as f:
            f.write('This ia a summary of arc information.\n')
            f.write('Created by Yang Fan (yfan_cge@hust.edu.cn)\n')
            f.write('Record time: %s \n\n' % datetime.datetime.now())
            f.write('Number of Arcs: %s \n' % len(self.__arcs))

            f.write('-------------------Please see more details in below:---------------------- \n\n')

            i = 0
            for arc in self.__arcs:
                f.write('Arc NO: %s;  %s ====> %s ; Length: %s \n' % (i, arc[0], arc[-1], len(arc)))
                i += 1

        return self

    def makeArcTF(self):
        rd = ArcData(interfaceConfig=self.__gf.L1b.InterfaceConfig)

        date = self.__gf.getDate()
        if not os.path.exists(self.__arcft_path):
            os.mkdir(self.__arcft_path)
        path = self.__arcft_path + '/' + date[0] + '_' + date[-1] + '.txt'
        with open(path, 'w') as f:
            f.write('This ia a summary of arc True or False information.\n')
            f.write('Created by Yi Wu (wu_yi@hust.edu.cn)\n')
            f.write('Record time: %s \n\n' % datetime.datetime.now())
            f.write('Number of Arcs: %s \n' % len(self.__arcs))

            f.write('-------------------Please see more details in below:---------------------- \n\n')

            i = 0
            for arc in self.__arcs:
                state = True
                info = ''
                GNVA = rd.getData(arc=i, kind=Payload.GNV, sat=SatID.A)
                GNVB = rd.getData(arc=i, kind=Payload.GNV, sat=SatID.B)
                kbrr = rd.getData(arc=i, kind=Payload.KBR, sat=SatID.A)
                if len(GNVB) != len(GNVA):
                    state = False
                    info = 'length GNVA != GNVB'
                if len(kbrr) == 0:
                    state = False
                    info = 'kbrr is None'
                f.write('Arc NO: %s %s ----- %s;\n' % (i, state, info))
                i += 1
        pass

    def rms(self, x: np.ndarray):
        """
        :param x: 1-dim
        :return:
        """
        # 确认公式
        return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])

    @staticmethod
    def divideArcByBool(arrayBool):
        """
        auto-generate arcs by bool evaluation. gaps are fill by False.
        :param arrayBool:
        :return: the index of True
        """
        arclist = []

        trueIndex = np.array(arrayBool.nonzero()).flatten()

        a = trueIndex[0:-1]
        b = trueIndex[1:]

        diff = b - a
        if (diff == 1).all():
            '''No gap'''
            arclist.append(trueIndex)
            return arclist

        diffIndex = np.where(diff > 1)[0]
        arclist.append(trueIndex[np.arange(0, diffIndex[0] + 1)])

        for i in range(len(diffIndex) - 1):
            arclist.append(trueIndex[np.arange(diffIndex[i] + 1, diffIndex[i + 1] + 1)])

        arclist.append(trueIndex[np.arange(diffIndex[-1] + 1, len(trueIndex))])

        return arclist

    def __outlier(self, GNV, KiOrb):
        GNV_time = GNV[:, 0]
        KiOrb_time = KiOrb[:, 0]
        int_t_kin = KiOrb_time.astype(np.int64)
        int_t_gnv = GNV_time.astype(np.int64)
        new_index = [list(int_t_gnv).index(x) for x in int_t_kin]
        GNV_time = GNV_time[new_index]
        GNV = GNV[new_index, :]
        '''Outlier'''
        diff = KiOrb[:, 1:] - GNV[:, 1:4]
        index = Outlier(rms_times=self.__OrbitConfig.OutlierTimes, upper_RMS=self.__OrbitConfig.OutlierLimit).remove_V2(
            GNV_time, diff[:, 0])
        return KiOrb[index, :]


class ArcData:

    def __init__(self, interfaceConfig: InterfaceConfig):
        self.date_span = interfaceConfig.date_span
        self.InterfaceConfig = interfaceConfig
        self.dir = pathlib.Path()
        self.InterfacePathConfig = InterfaceConfig.PathOfFiles(self.InterfaceConfig)
        self.InterfacePathConfig.__dict__.update(self.InterfaceConfig.PathOfFilesConfig.copy())
        raw_data_path = self.InterfacePathConfig.temp_raw_data
        filename = self.dir.joinpath(raw_data_path, self.date_span[0] + '_' + self.date_span[1] + ".hdf5")
        self.__h5data = h5py.File(filename, "r")
        pass

    def getData(self, arc: int, kind: Payload, sat=SatID.A):
        if isinstance(sat, str):
            spec = kind.name + '_' + sat + '_arc' + str(arc)
        else:
            spec = kind.name + '_' + sat.name + '_arc' + str(arc)
        return self.__h5data[spec][()]

    def closeH5py(self):
        self.__h5data.close()
        pass

