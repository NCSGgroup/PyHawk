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
import matplotlib
from datetime import datetime
import datetime
import calendar
plt.rcParams['font.sans-serif']=['Microsoft YaHei']         #指定默认字体（因为matplotlib默认为英文字体，汉字会使其乱码）
plt.rcParams['axes.unicode_minus']=False    #可显示‘-’负号


def GRACE_FO(data):
    '''static model'''
    static_c, static_s = LoadGif48().load('../data/StaticGravityField/GGM05C.gfc').getCS(60)
    '''read hust'''
    hust_filename = '../result/SH/GFO/HAWK01/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + data + '_GRAC_HUST_BA01_0600.gfc'
    hust_c, hust_s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
    hust_c, hust_s = hust_c - static_c, hust_s - static_s
    hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
    '''read CSR 06'''
    csr_06_name = '../result/SH/GFO/CSR/GSM-2_'+data+'_GRFO_UTCSR_BA01_0600.gfc'
    csr_06_c, csr_06_s = LoadCSR().load(fileIn=csr_06_name).getCS(Nmax=60)
    csr_06_c, csr_06_s = csr_06_c - static_c, csr_06_s - static_s
    csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_06_c), GeoMathKit.CS_1dTo2d(csr_06_s)
    '''read JPL 06'''
    jpl_filename = '../result/SH/GFO/JPL/GSM-2_'+data+'_GRFO_JPLEM_BA01_0600.gfc'
    jpl_c, jpl_s = LoadCSR().load(fileIn=jpl_filename).getCS(Nmax=60)
    jpl_c, jpl_s = jpl_c - static_c, jpl_s - static_s
    jpl_c, jpl_s = GeoMathKit.CS_1dTo2d(jpl_c), GeoMathKit.CS_1dTo2d(jpl_s)
    '''read GFZ 06'''
    gfz_filename = '../result/SH/GFO/GFZ/GSM-2_'+data+'_GRFO_GFZOP_BA01_0600.gfc'
    gfz_c, gfz_s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
    gfz_c, gfz_s = gfz_c - static_c, gfz_s - static_s
    gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)

    '''hust'''
    hust = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    '''csr 06'''
    csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
    '''gfz 06'''
    gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
    '''jpl 06'''
    jpl = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)
    plt.title('Geoid Degree Error RMS(2019-01)', fontsize=24)
    plt.plot(np.arange(2, 61, 1), hust[2:], color='green', marker='^', label='HAWK 06')
    plt.plot(np.arange(2, 61, 1), csr[2:], color='red', marker='.', label='CSR 06')
    plt.plot(np.arange(2, 61, 1), gfz[2:], color='black', marker='s', label='GFZ 06')
    plt.plot(np.arange(2, 61, 1), jpl[2:], color='blue', marker='p', label='JPL 06')
    plt.xlim(0, 60)
    plt.ylim(1e-5, 1e-2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Degree', fontsize=24)
    plt.ylabel('Degree Error RMS', fontsize=24)
    plt.yscale("log")
    plt.legend(fontsize=20, loc=3)
    plt.grid(True, which='both', linestyle='-.')
    plt.show()
    pass


def drawTime(data1, data2):
    '''static model'''
    static_c, static_s = LoadGif48().load('../data/StaticGravityField/gif48.gfc').getCS(60)
    '''read hust'''
    hust_filename = '../result/SH/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + data1 + '_GRAC_HUST_BA01_0600.gfc'
    hust_c, hust_s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
    hust_c, hust_s = hust_c - static_c, hust_s - static_s
    # hust_c, hust_s = hust_c - static_c, hust_s - static_s
    hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
    '''read hust'''
    hust_filename2 = '../result/CRA_Licom_1007/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + data1 + '_GRAC_HUST_BA01_0600.gfc'
    hust_c2, hust_s2 = LoadCSR().load(fileIn=hust_filename2).getCS(Nmax=60)
    hust_c2, hust_s2 = hust_c2 - static_c, hust_s2 - static_s
    hust_c2, hust_s2 = GeoMathKit.CS_1dTo2d(hust_c2), GeoMathKit.CS_1dTo2d(hust_s2)
    '''read hust'''
    hust_filename3 = '../result/SH/RL06/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_2006-01-05-2006-02-04_GRAC_HUST_BA01_0600.gfc'
    hust_c3, hust_s3 = LoadCSR().load(fileIn=hust_filename3).getCS(Nmax=60)
    hust_c3, hust_s3 = hust_c3 - static_c, hust_s3 - static_s
    hust_c3, hust_s3 = GeoMathKit.CS_1dTo2d(hust_c3), GeoMathKit.CS_1dTo2d(hust_s3)
    '''read hust'''
    hust_filename4 = '../result/SH/RL06/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_2009-01-01-2009-01-31_GRAC_HUST_BA01_0600.gfc'
    hust_c4, hust_s4 = LoadCSR().load(fileIn=hust_filename4).getCS(Nmax=60)
    hust_c4, hust_s4 = hust_c4 - static_c, hust_s4 - static_s
    hust_c4, hust_s4 = GeoMathKit.CS_1dTo2d(hust_c4), GeoMathKit.CS_1dTo2d(hust_s4)
    '''read CSR 05'''
    csr_05_name = '../result/SH/CSR/GSM-2_2006001-2006031_GRAC_UTCSR_BA01_0600.gfc'
    csr_05_c, csr_05_s = LoadCSR().load(fileIn=csr_05_name).getCS(Nmax=60)
    csr_05_c, csr_05_s = csr_05_c - static_c, csr_05_s - static_s
    csr_05_c, csr_05_s = GeoMathKit.CS_1dTo2d(csr_05_c), GeoMathKit.CS_1dTo2d(csr_05_s)
    '''read CSR'''
    csr_filename = '../result/SH/CSR/GSM-2_' + data2 + '_GRAC_UTCSR_BA01_0600.gfc'
    csr_c, csr_s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
    csr_c, csr_s = csr_c - static_c, csr_s - static_s
    csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_c), GeoMathKit.CS_1dTo2d(csr_s)
    # csr_diff_c = hust_c - csr_c
    # csr_diff_s = hust_s - csr_s
    # csr_e_l = get_drgee_rms(diff_c=csr_diff_c, diff_s=csr_diff_s)
    # csr_n_l = get_geoid_drgee_error(diff_c=csr_diff_c, diff_s=csr_diff_s)
    '''read GFZ'''
    gfz_filename = '../result/SH/GFZ/GSM-2_' + data2 + '_GRAC_GFZOP_BA01_0600.gfc'
    gfz_c, gfz_s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
    gfz_c, gfz_s = gfz_c - static_c, gfz_s - static_s
    gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)
    '''read GFZ 05'''
    gfz_05_name = '../compare/GFZ_GFZ-Release-05_monthly_unfiltered_GSM-2_2008001-2008031_0031_EIGEN_G---_005a.gfc'
    gfz_05_c, gfz_05_s = LoadCSR().load(fileIn=gfz_05_name).getCS(Nmax=60)
    gfz_05_c, gfz_05_s = gfz_05_c - static_c, gfz_05_s - static_s
    gfz_05_c, gfz_05_s = GeoMathKit.CS_1dTo2d(gfz_05_c), GeoMathKit.CS_1dTo2d(gfz_05_s)
    # gfz_diff_c = hust_c - gfz_c
    # gfz_diff_s = hust_s - gfz_s
    # gfz_e_l = get_drgee_rms(diff_c=gfz_diff_c, diff_s=gfz_diff_s)
    # gfz_n_l = get_geoid_drgee_error(diff_c=gfz_diff_c, diff_s=gfz_diff_s)
    '''read JPL'''
    jpl_filename = '../result/SH/JPL/GSM-2_' + data2 + '_GRAC_JPLEM_BA01_0600.gfc'
    jpl_c, jpl_s = LoadCSR().load(fileIn=jpl_filename).getCS(Nmax=60)
    jpl_c, jpl_s = jpl_c - static_c, jpl_s - static_s
    jpl_c, jpl_s = GeoMathKit.CS_1dTo2d(jpl_c), GeoMathKit.CS_1dTo2d(jpl_s)
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
    hust2 = get_geoid_drgee_error(diff_c=hust_c2, diff_s=hust_s2)
    hust3 = get_geoid_drgee_error(diff_c=hust_c3, diff_s=hust_s3)
    hust4 = get_geoid_drgee_error(diff_c=hust_c4, diff_s=hust_s4)
    static = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    '''csr'''
    csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
    '''csr 05'''
    csr_05 = get_geoid_drgee_error(diff_c=csr_05_c, diff_s=csr_05_s)
    '''gfz'''
    gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
    '''gfz 05'''
    gfz_05 = get_geoid_drgee_error(diff_c=gfz_05_c, diff_s=gfz_05_s)
    '''jpl'''
    jpl = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)
    # plt.title('Geoid Degree', fontsize=30)
    matplotlib.rcParams.update({'font.size': 30})
    fig = plt.figure(figsize=(10, 8))
    plt.style.use(['science'])
    plt.plot(np.arange(2, 61, 1), hust[2:], color='green', linewidth=2.0, marker='', label='PyHawk')
    # plt.plot(np.arange(2, 61, 1), hust2[2:], color='black', linewidth=2.0, marker='', label='PyHawk-CRA')
    # plt.plot(np.arange(2, 61, 1), hust3[2:], color='olive', linewidth=2.0, marker='', label='2006-01-04-2006-02-03')

    # plt.plot(np.arange(2, 61, 1), hust4[2:], color='olive', marker='^', label='HUST kin-3h-2e11')
    plt.plot(np.arange(2, 61, 1), csr[2:], color='red', marker='', linewidth=2.0,label='CSR')
    plt.plot(np.arange(2, 61, 1), gfz[2:], color='indigo', marker='', linewidth=2.0,label='GFZ')
    plt.plot(np.arange(2, 61, 1), jpl[2:], color='blue', marker='',linewidth=2.0, label='JPL')
    # plt.plot(np.arange(2, 61, 1), csr_05[2:], color='indigo', marker='h', label='CSR 05')
    # plt.plot(np.arange(2, 61, 1), gfz_05[2:], color='blue', marker='H', label='GFZ 05')
    # plt.plot(np.arange(2, 61, 1), csr_e_l[2:], color='red', marker='s', label='CSR - HUST')
    # plt.plot(np.arange(2, 61, 1), gfz_e_l[2:], color='blue', marker='p', label='GFZ - HUST')
    # plt.plot(np.arange(2, 61, 1), jpl_e_l[2:], color='green', marker='P', label='JPL - HUST')
    # plt.plot(np.arange(2, 61, 1), csr_gfz_e_l[2:], color='black', marker='*', label='GFZ - CSR')
    # plt.plot(np.arange(2, 61, 1), csr_jpl_e_l[2:], color='indigo', marker='h', label='JPL - CSR')
    # plt.plot(np.arange(2, 61, 1), gfz_jpl_e_l[2:], color='tan', marker='H', label='JPL - GFZ')
    plt.xlim(0, 60)
    plt.ylim(1e-5, 1e-2)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('Degree', fontsize=28)
    plt.ylabel('Degree geoid height [m]', fontsize=28)
    plt.yscale("log")
    plt.legend(fontsize=26, loc=1)
    plt.grid(True, which='both', linestyle='-.')
    # plt.savefig('../result/img/CRA_LICOM_v2/' + data1 + '.png')
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


def drawKBRR():
    C0C1h5 = h5py.File('C0C1RangeRate_0.hdf5', 'r')
    C0C1C2h5 = h5py.File('C0C1C2_RangeRate_0.hdf5', 'r')
    Testh5 = h5py.File('Test_RangeRate_0.hdf5', 'r')
    C0C1_45_h5 = h5py.File('45min_RangeRate_0.hdf5', 'r')
    print(rms(C0C1_45_h5['post_residual'][:]))
    print(rms(C0C1h5['post_residual'][:]))
    print(rms(C0C1C2h5['post_residual'][:]))
    print(rms(Testh5['post_residual'][:]))
    plt.title('diff adjust method(2008-1-Arc0)', fontsize=24)
    plt.plot(C0C1_45_h5['post_t'][:], C0C1_45_h5['post_residual'][:], color='red', marker='p', label='a+bt + ct^2 + (E+Ft)cosnt+(G+Ht)sinnt(t=45)')
    plt.plot(C0C1h5['post_t'][:], C0C1h5['post_residual'][:], color='peru', marker='^', label='a + bt + (E+Ft)cosnt+(G+Ht)sinnt')
    plt.plot(C0C1C2h5['post_t'][:], C0C1C2h5['post_residual'][:], color='indigo', marker='h', label='a+bt + ct^2 + (E+Ft)cosnt+(G+Ht)sinnt')
    plt.plot(Testh5['post_t'][:], Testh5['post_residual'][:], color='olive', marker='.', label='a+bt + ct^2 + dt^3 + (E+Ft+Qt^2)cosnt+(G+Ht+F^2)sinnt')
    plt.xlabel('GPS Time', fontsize=24)
    plt.ylabel('m/s', fontsize=24)
    plt.legend(fontsize=20, loc=3)
    plt.grid(True, which='both', linestyle='-.')
    plt.show()


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


def GeoidGMT():

    pass

config = {
    "font.family": 'serif',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
matplotlib.rcParams.update(config)


def drawMean():
    '''static model'''
    static_c, static_s = LoadGif48().load('../data/StaticGravityField/gif48.gfc').getCS(60)
    '''read hust'''
    filenamelist = os.listdir(r'../result/SH/mean/AOD')
    hust_c, hust_s = 0, 0
    i = 0
    for filename in filenamelist:
        hust_filename = '../result/SH/mean/AOD/' + filename
        c, s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
        hust_c += c
        hust_s += s
        i += 1
    hust_c = hust_c / i
    hust_s = hust_s / i
    hust_c, hust_s = hust_c - static_c, hust_s - static_s
    hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
    '''read CSR'''
    filenamelist = os.listdir(r'../result/SH/mean/CRA')
    cra_c, cra_s = 0, 0
    j = 0
    for filename in filenamelist:
        cra_filename = '../result/SH/mean/CRA/' + filename
        c, s = LoadCSR().load(fileIn=cra_filename).getCS(Nmax=60)
        cra_c += c
        cra_s += s
        j += 1
    cra_c = cra_c / j
    cra_s = cra_s / j
    cra_c, cra_s = cra_c - static_c, cra_s - static_s
    cra_c, cra_s = GeoMathKit.CS_1dTo2d(cra_c), GeoMathKit.CS_1dTo2d(cra_s)
    # csr_diff_c = hust_c - csr_c
    # csr_diff_s = hust_s - csr_s
    # csr_e_l = get_drgee_rms(diff_c=csr_diff_c, diff_s=csr_diff_s)
    # csr_n_l = get_geoid_drgee_error(diff_c=csr_diff_c, diff_s=csr_diff_s)
    '''read CSR'''
    filenamelist = os.listdir(r'../result/SH/mean/CSR')
    csr_c, csr_s = 0, 0
    j = 0
    for filename in filenamelist:
        csr_filename = '../result/SH/mean/CSR/' + filename
        c, s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
        csr_c += c
        csr_s += s
        j += 1
    csr_c = csr_c / j
    csr_s = csr_s / j
    csr_c, csr_s = csr_c - static_c, csr_s - static_s
    csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_c), GeoMathKit.CS_1dTo2d(csr_s)
    '''read GFZ'''
    filenamelist = os.listdir(r'../result/SH/mean/GFZ')
    gfz_c, gfz_s = 0, 0
    for filename in filenamelist:
        gfz_filename = '../result/SH/mean/GFZ/' + filename
        c, s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
        gfz_c += c
        gfz_s += s
    gfz_c = gfz_c / 60
    gfz_s = gfz_s / 60
    gfz_c, gfz_s = gfz_c - static_c, gfz_s - static_s
    gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)
    '''read JPL'''
    filenamelist = os.listdir(r'../result/SH/mean/JPL')
    jpl_c, jpl_s = 0, 0
    for filename in filenamelist:
        jpl_filename = '../result/SH/mean/JPL/' + filename
        c, s = LoadCSR().load(fileIn=jpl_filename).getCS(Nmax=60)
        jpl_c += c
        jpl_s += s
    jpl_c = jpl_c / 60
    jpl_s = jpl_s / 60
    jpl_c, jpl_s = jpl_c - static_c, jpl_s - static_s
    jpl_c, jpl_s = GeoMathKit.CS_1dTo2d(jpl_c), GeoMathKit.CS_1dTo2d(jpl_s)
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
    hust = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    '''cra'''
    cra = get_geoid_drgee_error(diff_c=cra_c, diff_s=cra_s)
    '''csr'''
    csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
    '''gfz'''
    gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
    '''jpl'''
    jpl = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)
    # plt.title('Geoid Degree', fontsize=30)
    # matplotlib.rcParams.update({'font.size': 30})
    # plt.style.use(['science'])
    plt.plot(np.arange(2, 61, 1), hust[2:], color='green', linewidth=2.0, marker='', label='AOD')
    # plt.plot(np.arange(2, 61, 1), cra[2:], color='red', marker='', linewidth=2.0, label='CRA')
    plt.plot(np.arange(2, 61, 1), gfz[2:], color='black', marker='', linewidth=2.0, label='GFZ')
    plt.plot(np.arange(2, 61, 1), jpl[2:], color='blue', marker='', linewidth=2.0, label='JPL')
    plt.plot(np.arange(2, 61, 1), csr[2:], color='red', marker='', linewidth=2.0, label='CSR')
    plt.xlim(0, 60)
    plt.ylim(1e-5, 1e-2)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('Degree', fontsize=28)
    plt.ylabel('Degree geoid height [m]', fontsize=28)
    plt.yscale("log")
    plt.legend(fontsize=26, loc=1)
    plt.grid(True, which='both', linestyle='-.')
    plt.show()


def get_month_start_and_end_by_date_interval(start_date, end_date):
    """
    获取指定时间段内每个月的起止时间参数列表
    :param start_date: 起始时间 --> str
    :param end_date: 结束时间 --> str
    :return: date_range_list -->list [{'2019-10-01': ['2019-10-01', '2019-10-31']}, ...}]

    """
    date_range_list = []
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    # 计算开始日期的当月起止日期
    start_month_first = datetime.datetime(year=start_date.year, month=start_date.month, day=1).date()
    start_month_last = datetime.datetime(year=start_date.year, month=start_date.month,
                                         day=calendar.monthrange(start_date.year, start_date.month)[1]).date()
    date_range_list.append([start_month_first.strftime('%Y-%m-%d'),start_month_last.strftime('%Y-%m-%d')])

    # 计算结束日期的当月起止日期
    end_month_first = datetime.datetime(year=end_date.year, month=end_date.month, day=1).date()
    end_month_last = datetime.datetime(year=end_date.year, month=end_date.month,
                                       day=calendar.monthrange(end_date.year, end_date.month)[1]).date()

    while True:
        # 下个月开始日期
        next_month_start = datetime.datetime(year=start_date.year, month=start_date.month,
                                             day=calendar.monthrange(start_date.year, start_date.month)[
                                                 1]).date() + datetime.timedelta(days=1)
        next_month_end = datetime.datetime(year=next_month_start.year, month=next_month_start.month,
                                           day=calendar.monthrange(next_month_start.year, next_month_start.month)[
                                               1]).date()
        if next_month_start < end_month_first:
            date_range_list.append([next_month_start.strftime('%Y-%m-%d'),next_month_end.strftime('%Y-%m-%d')])
            start_date = next_month_start
        else:
            break
    # 避免传入的起止日期在同一月导致数据重复
    temp_dict = [end_month_first.strftime('%Y-%m-%d'), end_month_last.strftime('%Y-%m-%d')]
    if temp_dict not in date_range_list:
        date_range_list.append(temp_dict)
    return date_range_list


def datetransfor(ymd):
    y, m, d = map(int, ymd.split("-"))
    date = datetime.datetime(y, m, d)
    day_delta = date - datetime.datetime(y, 1, 1) + datetime.timedelta(1)
    yday = str(y) + str(day_delta.days).rjust(3, "0")
    return yday


def subplotsGeoid():
    matplotlib.rcParams.update({'font.size': 26})
    plt.style.use(['science'])
    fig, ax = plt.subplots(3, 4, figsize=(14, 9.6), sharex=True, sharey=True)
    data_span = get_month_start_and_end_by_date_interval('2007-01-01', '2007-12-31')
    '''static model'''
    static_c, static_s = LoadGif48().load('../data/StaticGravityField/gif48.gfc').getCS(60)
    for i in range(len(data_span)):
        data1 = data_span[i][0] + '-' + data_span[i][1]
        data2 = datetransfor(data_span[i][0]) + '-' + datetransfor(data_span[i][1])
        month = data_span[i][0].split('-')[0] + '-' + data_span[i][0].split('-')[1]
        plt.subplot(3, 4, i+1)
        '''read hust'''
        hust_filename = '../result/SH/RL06/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + data1 + '_GRAC_HUST_BA01_0600.gfc'
        hust_c, hust_s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
        hust_c, hust_s = hust_c - static_c, hust_s - static_s
        hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
        '''read CSR'''
        csr_filename = '../result/SH/CSR/GSM-2_' + data2 + '_GRAC_UTCSR_BA01_0600.gfc'
        csr_c, csr_s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
        csr_c, csr_s = csr_c - static_c, csr_s - static_s
        csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_c), GeoMathKit.CS_1dTo2d(csr_s)
        '''read GFZ'''
        gfz_filename = '../result/SH/GFZ/GSM-2_' + data2 + '_GRAC_GFZOP_BA01_0600.gfc'
        gfz_c, gfz_s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
        gfz_c, gfz_s = gfz_c - static_c, gfz_s - static_s
        gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)
        '''read JPL'''
        jpl_filename = '../result/SH/JPL/GSM-2_' + data2 + '_GRAC_JPLEM_BA01_0600.gfc'
        jpl_c, jpl_s = LoadCSR().load(fileIn=jpl_filename).getCS(Nmax=60)
        jpl_c, jpl_s = jpl_c - static_c, jpl_s - static_s
        jpl_c, jpl_s = GeoMathKit.CS_1dTo2d(jpl_c), GeoMathKit.CS_1dTo2d(jpl_s)
        hust = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
        csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
        gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
        jpl = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)
        plt.style.use(['science'])
        plt.yscale("log")
        plt.title(month, fontsize=22)
        plt.plot(np.arange(2, 61, 1), hust[2:], color='green', linewidth=2.0, marker='', label='PyHawk')
        plt.plot(np.arange(2, 61, 1), csr[2:], color='red', marker='', linewidth=2.0, label='CSR')
        plt.plot(np.arange(2, 61, 1), gfz[2:], color='black', marker='', linewidth=2.0, label='GFZ')
        plt.plot(np.arange(2, 61, 1), jpl[2:], color='blue', marker='', linewidth=2.0, label='JPL')
        plt.xlim(0, 60)
        plt.xticks([0, 10, 20, 30, 40, 50, 60], ['0', '10', '20', '30', '40', '50', '60'], fontsize=20)
        plt.ylim(1e-5, 1e-2)
        plt.yticks([1e-5, 1e-4, 1e-3, 1e-2], [r'$\mathrm{1e}^{-5}$', r'$\mathrm{1e}^{-4}$',
                                              r'$\mathrm{1e}^{-3}$', r'$\mathrm{1e}^{-2}$'], fontsize=20)
        border = plt.gca()
        border.spines['top'].set_linewidth(1)  # 设置顶部边框线宽度为2
        border.spines['right'].set_linewidth(1)  # 设置右侧边框线宽度为2
        border.spines['bottom'].set_linewidth(1)  # 设置底部边框线宽度为2
        border.spines['left'].set_linewidth(1)  # 设置左侧边框线宽度为2
        if i == 9:

            leg = plt.legend(ncol=5, bbox_to_anchor=(0.9, -0.27), loc='upper center', frameon=True)
            for line in leg.get_lines():
                line.set_linewidth(4.0)
        if i == 4:
            plt.ylabel('Degree geoid height [m]', fontsize=28)
        # plt.legend(fontsize=26, loc=1)
        plt.grid(True, which='both', linestyle='-.')
    fig.text(0.508, 0.085, 'Degree', fontsize=24)
    # plt.xlabel('Degree', fontsize=24, horizontalalignment='right')
    plt.subplots_adjust(left=0.09, right=0.981, top=0.94, bottom=0.15, wspace=0.1, hspace=0.27)
    plt.savefig('../result/img/2007/sh.pdf')
    plt.show()


    pass


if __name__ == '__main__':
    span1 = '2010-10-01-2010-10-31'
    span2 = '2010274-2010304'
    data_span = get_month_start_and_end_by_date_interval('2009-01-01', '2010-12-31')

    # for i in range(len(data_span)):
    #     span1 = data_span[i][0] + '-' + data_span[i][1]
    #     span2 = datetransfor(data_span[i][0]) + '-' + datetransfor(data_span[i][1])
    #     drawTime(data1=span1, data2=span2)
    #     # GRACE_FO(data=span)
    #     # GeoidGMT()
    #     # drawPotentialError()
    #     # drawEWH()
    #     # drawMat()
    #     # drawKBRR()
    #     # drawMean()
    drawTime(data1=span1, data2=span2)
    # subplotsGeoid()
    pass


