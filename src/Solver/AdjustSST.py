import pathlib

import h5py
import src.Preference.Pre_AdjustOrbit as Pre_AdjustOrbit
from src.Interface.ArcOperation import ArcData
import src.Preference.Pre_Interface as Pre_Interface
import src.Preference.EnumType as EnumType


class RangeRate:

    def __init__(self, arcNo: int, date_span=Pre_Interface.date_span):

        filename = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp).\
            joinpath(date_span[0] + '_' + date_span[1]).joinpath(str(arcNo) + '_AB.hdf5')
        h5AB = h5py.File(filename, 'r')

        rd = ArcData(date_span=date_span)
        KBR = rd.getData(arc=arcNo, kind=EnumType.Payload.KBR)
        pass