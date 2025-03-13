# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2025/3/13 18:25
# @File    : demo_DataCollection.py
# @Software: PyCharm

import sys
sys.path.append('../')
from src.Auxilary.GeoMathKit import GeoMathKit
import os
import wget
import json
from src.Preference.EnumType import Level, Mission, SatID


class DataCollection:

    def __init__(self, config: dict):

        self.Misson = Mission[config['Mission']]
        self.Level = Level[config['Level']]
        # self.Sat = SatID[config['Sat']]
        # self._LocalDir = os.path.join(config['Local directory'], self.Level.name)
        self._LocalDir = config['Local directory']
        self._RemoteDir = config['Remote site']
        self.dataspan = [config['Data begin'], config['Data end']]

        self.byDay(*self.dataspan)

    def downLoadFile(self, LocalFile, RemoteFile):
        """
        single file download
        :param LocalFile:
        :param RemoteFile:
        :return:
        """
        wget.download(RemoteFile, LocalFile)
        return True

    def byDay(self, begin, end):
        """
        usage: begin='2002-01-01', end='2002-02-03'
        :param begin:
        :param end:
        :return:
        """
        daylist = GeoMathKit.dayListByDay(begin, end)

        for day in daylist:

            localdir = os.path.join(self._LocalDir, day.strftime("%Y-%m"))
            isExists = os.path.exists(localdir)
            if not isExists:
                print('Month for %s' % day.strftime("%Y-%m"))
                os.makedirs(localdir)

            outname, RemotePath = None, None
            if self.Misson is Mission.GRACE_RL02:
                outname = 'grace_%s_%s_02.tar.gz' % (self.Level.name[1:3], day.strftime("%Y-%m-%d"))
                RemotePath = os.path.join(self._RemoteDir, day.strftime("%Y/"), outname)
            if self.Misson is Mission.GRACE_RL03:
                outname = 'grace_%s_%s_03.tar.gz' % (self.Level.name[1:3], day.strftime("%Y-%m"))
                RemotePath = os.path.join(self._RemoteDir, outname)
            # elif self.Misson is Mission.AOD:
            #     outname = 'AOD%s_%s_X_06.asc.gz' % (self.Level.name[1:3], day.strftime("%Y-%m-%d"))
            #     RemotePath = os.path.join(self._RemoteDir, day.strftime("%Y/"), outname)
            elif self.Misson is Mission.GRACE_FO_RL04:
                outname = 'gracefo_%s_%s_RL04.ascii.noLRI.tgz' % (self.Level.name[1:3], day.strftime("%Y-%m-%d"))
                RemotePath = os.path.join(self._RemoteDir, day.strftime("%Y/"), outname)
            # elif self.Misson is Mission.Oribit:
            #     outname = 'Grace%s-kinematicOrbit-%s.tar.gz' % (self.Sat.name, day.strftime("%Y-%m-%d"))
            #     RemotePath = os.path.join(self._RemoteDir, day.strftime("%Y/"), outname)

            else:
                pass

            print('\n' + outname)
            print(os.path.join(localdir, outname))
            print(RemotePath)
            self.downLoadFile(LocalFile=localdir, RemoteFile=RemotePath)

            if self.Misson is Mission.GRACE_FO_RL04:
                GeoMathKit.un_tar(os.path.join(localdir, outname))
            # elif self.Misson is Mission.Oribit:
            #     GeoMathKit.un_targz(os.path.join(localdir, outname))
            elif self.Misson is Mission.GRACE_RL02:
                GeoMathKit.un_targz(os.path.join(localdir, outname))
            else:
                GeoMathKit.un_gz(os.path.join(localdir, outname))

        pass

    @staticmethod
    def defaultConfig():
        config = {}

        config['Local directory'] = './RL06/'
        config['Remote site'] = 'ftp://anonymous@isdcftp.gfz-potsdam.de/grace-fo/Level-1B/GFZ/AOD/RL06/'
        config['Data begin'] = '2021-01'
        config['Data end'] = '2021-12'

        with open('Config.json', 'w') as f:
            f.write(json.dumps(config))
        pass


def demo(config='../setting/demo_DataCollection/GRACE_AOD.download.json'):
    """
    download official RL06
    :return:
    """

    with open(config) as ff:
        configJob = json.load(ff)
    dc = DataCollection(config=configJob)
    pass


if __name__ == '__main__':
    # demo(config='GRACE_FO_L1A.download.json')
    # demo(config='GRACE_FO_L1B.download.json')
    demo(config='../setting/demo_DataCollection/GRACE_L1B.download.json')
    # demo(config='GRACE_Oribit.download.json')
    # demo(config='GRACE_AOD.download.json')
    # demo(config='GRACE_AOD.download.json')
