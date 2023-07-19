import h5py
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Interface.LoadSH import LoadCSR
from src.Auxilary.SortCS import SortCS
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Auxilary.LoveNumber import LoveNumber
from src.ConvertFieldType.ConvertSHC import ConvertSHC
from src.Preference.EnumType import FieldType
from src.Interface.LoadSH import LoadGif48
plt.rcParams['font.sans-serif']=['Microsoft YaHei']         #指定默认字体（因为matplotlib默认为英文字体，汉字会使其乱码）
plt.rcParams['axes.unicode_minus']=False    #可显示‘-’负号


def drawTime():
    '''static model'''
    static_c, static_s = LoadGif48().load('../data/StaticGravityField/gif48.gfc').getCS(60)
    '''read hust'''
    hust_filename = '../compare/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_2008-01-01-2008-01-31_GRAC_HUST_BA01_0600.gfc'
    hust_c, hust_s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
    hust_c, hust_s = hust_c - static_c, hust_s - static_s
    hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
    '''read CSR'''
    csr_filename = '../compare/CSR_CSR-Release-06_60x60_unfiltered_GSM-2_2008001-2008031_GRAC_UTCSR_BA01_0600.gfc'
    csr_c, csr_s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
    csr_c, csr_s = csr_c - static_c, csr_s - static_s
    csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_c), GeoMathKit.CS_1dTo2d(csr_s)
    # csr_diff_c = hust_c - csr_c
    # csr_diff_s = hust_s - csr_s
    # csr_e_l = get_drgee_rms(diff_c=csr_diff_c, diff_s=csr_diff_s)
    # csr_n_l = get_geoid_drgee_error(diff_c=csr_diff_c, diff_s=csr_diff_s)
    '''read GFZ'''
    gfz_filename = '../compare/GFZ_GFZ-Release-06_60x60_unfiltered_GSM-2_2008001-2008031_GRAC_GFZOP_BA01_0600.gfc'
    gfz_c, gfz_s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
    gfz_c, gfz_s = gfz_c - static_c, gfz_s - static_s
    gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)
    # gfz_diff_c = hust_c - gfz_c
    # gfz_diff_s = hust_s - gfz_s
    # gfz_e_l = get_drgee_rms(diff_c=gfz_diff_c, diff_s=gfz_diff_s)
    # gfz_n_l = get_geoid_drgee_error(diff_c=gfz_diff_c, diff_s=gfz_diff_s)
    # '''read JPL'''
    # jpl_filename = '../compare/JPL_JPL-Release-05_unfiltered_GSM-2_2008001-2008031_0031_JPLEM_0001_0005.gfc'
    # jpl_c, jpl_s = LoadCSR().load(fileIn=jpl_filename).getCS(Nmax=60)
    # jpl_c, jpl_s = GeoMathKit.CS_1dTo2d(jpl_c), GeoMathKit.CS_1dTo2d(jpl_s)
    # jpl_diff_c = hust_c - jpl_c
    # jpl_diff_s = hust_s - jpl_s
    # jpl_e_l = get_drgee_rms(diff_c=jpl_diff_c, diff_s=jpl_diff_s)
    # jpl_n_l = get_geoid_drgee_error(diff_c=jpl_diff_c, diff_s=jpl_diff_s)
    # '''csr and gfz'''
    # csr_gfz_diff_c = gfz_c - csr_c
    # csr_gfz_diff_s = gfz_s - csr_s
    # csr_gfz_e_l = get_drgee_rms(diff_c=csr_gfz_diff_c, diff_s=csr_gfz_diff_s)
    # '''csr and jpl'''
    # csr_jpl_diff_c = jpl_c - csr_c
    # csr_jpl_diff_s = jpl_s - csr_s
    # csr_jpl_e_l = get_drgee_rms(diff_c=csr_jpl_diff_c, diff_s=csr_jpl_diff_s)
    # '''gfz and jpl'''
    # gfz_jpl_diff_c = gfz_c - jpl_c
    # gfz_jpl_diff_s = gfz_s - jpl_s
    # gfz_jpl_e_l = get_drgee_rms(diff_c=gfz_jpl_diff_c, diff_s=gfz_jpl_diff_s)
    '''hust'''
    # hust = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    hust = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    static = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    '''csr'''
    csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
    '''gfz'''
    gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
    plt.title('CSR GFZ JPL with HUST Degree Error RMS', fontsize=24)
    plt.plot(np.arange(2, 61, 1), hust[2:], color='peru', marker='^', label='HUST')
    plt.plot(np.arange(2, 61, 1), csr[2:], color='olive', marker='.', label='CSR')
    plt.plot(np.arange(2, 61, 1), gfz[2:], color='red', marker='s', label='GFZ')
    # plt.plot(np.arange(2, 61, 1), csr_e_l[2:], color='red', marker='s', label='CSR - HUST')
    # plt.plot(np.arange(2, 61, 1), gfz_e_l[2:], color='blue', marker='p', label='GFZ - HUST')
    # plt.plot(np.arange(2, 61, 1), jpl_e_l[2:], color='green', marker='P', label='JPL - HUST')
    # plt.plot(np.arange(2, 61, 1), csr_gfz_e_l[2:], color='black', marker='*', label='GFZ - CSR')
    # plt.plot(np.arange(2, 61, 1), csr_jpl_e_l[2:], color='indigo', marker='h', label='JPL - CSR')
    # plt.plot(np.arange(2, 61, 1), gfz_jpl_e_l[2:], color='tan', marker='H', label='JPL - GFZ')
    plt.xlim(0, 60)
    plt.ylim(1e-5, 1e-2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Degree', fontsize=24)
    plt.ylabel('Degree Error RMS', fontsize=24)
    plt.yscale("log")
    plt.legend(fontsize=20, loc=10)
    plt.grid(True, which='both', linestyle='-.')
    plt.show()


def drawPotentialError():
    h5 = h5py.File('../compare/2008-01-01_2008-01-31.hdf5', 'r')
    my_c = h5['c'][:]
    my_s = h5['s'][:]
    my_c[0, :] = 0
    my_s[0, :] = 0
    my_c[1, :] = 0
    my_s[1, :] = 0
    my_c = np.fliplr(my_c)
    Potential = np.matrix(np.hstack((my_c, my_s)))
    # Potential[Potential == 0.0] = np.nan
    print(np.shape(Potential))
    print(Potential.max())
    print(Potential.min())
    plt.matshow(Potential, vmin=0, vmax=2.439348474467162e-06)
    plt.colorbar()
    plt.show()
    pass


def drawEWH():
    force_model_config = ForceModelConfig()
    fmDict = json.load(open('..//setting/Calibrate/ForceModelConfig.json', 'r'))
    force_model_config.__dict__ = fmDict
    h5 = h5py.File('../compare/2008-01-01_2008-01-31.hdf5', 'r')
    my_c = h5['c'][:]
    my_s = h5['s'][:]

    cqlm = my_c[np.newaxis, :]
    sqlm = my_s[np.newaxis, :]
    ln = LoveNumber().configure(FMConfig=force_model_config).getNumber()
    convert = ConvertSHC(cqlm, sqlm, FieldType.EWH, ln)
    convert.convert_to(FieldType.EWH)
    cqlm, sqlm = convert.cqlm, convert.sqlm
    print(1)
    pass


def get_drgee_rms(diff_c, diff_s):
    e_l = np.zeros(61)
    for l in range(0, 61):
        deta_c = 0
        deta_s = 0
        for m in range(0, l + 1):
            deta_c += np.power(diff_c[l][m], 2)
            deta_s += np.power(diff_s[l][m], 2)
        e_l[l] = np.sqrt((deta_c + deta_s) / (2 * l + 1))
    return e_l


def get_geoid_drgee_error(diff_c, diff_s):
    R = 6.3781363000E+06
    n_l = np.zeros(61)
    for l in range(0, 61):
        deta_c = 0
        deta_s = 0
        for m in range(0, l + 1):
            deta_c += np.power(diff_c[l][m], 2)
            deta_s += np.power(diff_s[l][m], 2)
        n_l[l] = R * np.sqrt(deta_c + deta_s)
    return n_l


def drawMat():
    hust_filename = '../compare/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_2008-01-01-2008-01-31_GRAC_HUST_BA01_0600.gfc'
    hust_c, hust_s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
    hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
    '''read CSR'''
    csr_filename = '../compare/CSR_CSR-Release-06_60x60_unfiltered_GSM-2_2008001-2008031_GRAC_UTCSR_BA01_0600.gfc'
    csr_c, csr_s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
    csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_c), GeoMathKit.CS_1dTo2d(csr_s)
    csr_diff_c = hust_c - csr_c
    csr_diff_s = hust_s - csr_s

    plt.matshow(csr_diff_c)
    plt.colorbar()
    plt.show()

    csr_e_l = get_drgee_rms(diff_c=csr_diff_c, diff_s=csr_diff_s)
    csr_n_l = get_geoid_drgee_error(diff_c=csr_diff_c, diff_s=csr_diff_s)
    '''read GFZ'''
    gfz_filename = '../compare/GFZ_GFZ-Release-06_60x60_unfiltered_GSM-2_2008001-2008031_GRAC_GFZOP_BA01_0600.gfc'
    gfz_c, gfz_s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
    gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)
    gfz_diff_c = hust_c - gfz_c
    gfz_diff_s = hust_s - gfz_s
    gfz_e_l = get_drgee_rms(diff_c=gfz_diff_c, diff_s=gfz_diff_s)
    gfz_n_l = get_geoid_drgee_error(diff_c=gfz_diff_c, diff_s=gfz_diff_s)
    '''read JPL'''
    jpl_filename = '../compare/JPL_JPL-Release-05_unfiltered_GSM-2_2008001-2008031_0031_JPLEM_0001_0005.gfc'
    jpl_c, jpl_s = LoadCSR().load(fileIn=jpl_filename).getCS(Nmax=60)
    jpl_c, jpl_s = GeoMathKit.CS_1dTo2d(jpl_c), GeoMathKit.CS_1dTo2d(jpl_s)
    jpl_diff_c = hust_c - jpl_c
    jpl_diff_s = hust_s - jpl_s
    jpl_e_l = get_drgee_rms(diff_c=jpl_diff_c, diff_s=jpl_diff_s)
    jpl_n_l = get_geoid_drgee_error(diff_c=jpl_diff_c, diff_s=jpl_diff_s)
    '''csr and gfz'''
    csr_gfz_diff_c = gfz_c - csr_c
    csr_gfz_diff_s = gfz_s - csr_s
    csr_gfz_e_l = get_drgee_rms(diff_c=csr_gfz_diff_c, diff_s=csr_gfz_diff_s)
    '''csr and jpl'''
    csr_jpl_diff_c = jpl_c - csr_c
    csr_jpl_diff_s = jpl_s - csr_s
    csr_jpl_e_l = get_drgee_rms(diff_c=csr_jpl_diff_c, diff_s=csr_jpl_diff_s)
    '''gfz and jpl'''
    gfz_jpl_diff_c = gfz_c - jpl_c
    gfz_jpl_diff_s = gfz_s - jpl_s
    gfz_jpl_e_l = get_drgee_rms(diff_c=gfz_jpl_diff_c, diff_s=gfz_jpl_diff_s)


if __name__ == '__main__':
    drawTime()
    # drawPotentialError()
    # drawEWH()
    # drawMat()