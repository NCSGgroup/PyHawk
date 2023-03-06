from src.Solver.NEQ import GravityNEQperArc
from src.Preference.EnumType import SatID
import src.Preference.Pre_NEQ as Pre_NEQ
import src.Preference.Pre_Interface as Pre_Interface
import pathlib
import h5py
import numpy as np


def demo():
    neq = GravityNEQperArc(0)
    neq.configure().get_neq()


def demo1():

    res_dir = pathlib.Path(Pre_NEQ.PathOfFiles.NormalEqTemp).joinpath\
        (Pre_Interface.date_span[0] + '_' + Pre_Interface.date_span[1])
    res_filename = res_dir.joinpath(str(Pre_Interface.arcNo) + '.hdf5')
    h5 = h5py.File(res_filename, 'r')

    print(h5.keys())
    print(np.shape(h5['Orbit_N_A'][:]))
    print(np.shape(h5['RangeRate_N'][:]))


if __name__ == '__main__':
    demo1()

