import h5py
import json
import os
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Interface.LoadSH import LoadCSR
from src.Auxilary.SortCS import SortCS
from src.Auxilary.GeoMathKit import GeoMathKit


def drawTime():
    h5 = h5py.File('../compare/2008-01-01_2008-01-31.hdf5', 'r')
    my_c = h5['c'][:]
    my_s = h5['s'][:]

    filename = '../compare/CSR_CSR-Release-06_60x60_unfiltered_GSM-2_2008001-2008031_GRAC_UTCSR_BA01_0600.gfc'
    ref_c, ref_s = LoadCSR().load(fileIn=filename).getCS(Nmax=60)

    my_c, my_s = GeoMathKit.CS_2dTo1d(my_c), GeoMathKit.CS_2dTo1d(my_s)

    diff_c = my_c - ref_c
    diff_s = my_s - ref_s

    print(my_c)
    print(ref_c)


if __name__ == '__main__':
    drawTime()
