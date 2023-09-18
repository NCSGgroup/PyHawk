import h5py

from src.Interface.Accelerometer import Accelerometer, AccCaliPar
from src.Interface.ArcOperation import ArcData
from src.Preference.EnumType import SatID, Payload
from src.Auxilary.GeoMathKit import GeoMathKit
from pathlib import Path
import numpy as np


def demo1():
    sat = SatID.B
    rd = ArcData()
    ap = AccCaliPar(sat=sat, arcNo=0)

    ac = Accelerometer(rd=rd, ap=ap)
    ac.get_and_save()
    res0 = ac.get_acceleration()
    res1 = ac.get_du2p()
    print(res0)
    pass


def analysis():
    data_span = ['2008-01-01', '2008-01-31']
    path = '../temp/NonConservativeForce'

    date_list = GeoMathKit.getEveryDay(*data_span)
    sat = 'A'
    for i in range(223):
        filename = Path(path, (data_span[0]+'_'+data_span[1]+'_'+sat+'_'+str(i)+'.hdf5'))
        h5 = h5py.File(filename, 'r')



if __name__ == '__main__':
    # demo1()
    analysis()