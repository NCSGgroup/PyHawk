import h5py
import json
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.Preference.Pre_Parameterization import ParameterConfig
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Interface.LoadSH import LoadCSR
from src.Auxilary.SortCS import SortCS
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Auxilary.LoveNumber import LoveNumber
from src.ConvertFieldType.ConvertSHC import ConvertSHC
from src.Preference.EnumType import FieldType
from src.Interface.LoadSH import LoadGif48, LoadGOCO06S
import matplotlib
from datetime import datetime


def GRACE(date):
    date1 = date.replace('_', '-')
    date2 = convert_range(date, standard_to_compact)
    '''static model'''
    static_c, static_s = LoadGif48().load('../data/StaticGravityField/gif48.gfc').getCS(60)
    '''read hust'''
    hust_filename = '../result/case-ESSD/Hawk_60x60_unfiltered_GSM-2_' + date1 + '_GRAC_0100.gfc'
    hust_c, hust_s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
    hust_c, hust_s = hust_c - static_c, hust_s - static_s
    hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
    # '''read hust'''
    hust_filename2 = '../result/case-ESSD/Hawk_60x60_unfiltered_GSM-2_' + date1 + '_GRAC_0100.gfc'
    hust_c2, hust_s2 = LoadCSR().load(fileIn=hust_filename2).getCS(Nmax=60)
    hust_c2, hust_s2 = hust_c2 - static_c, hust_s2 - static_s
    hust_c2, hust_s2 = GeoMathKit.CS_1dTo2d(hust_c2), GeoMathKit.CS_1dTo2d(hust_s2)
    '''read hust'''
    hust_filename3 = '../result/grace_v3/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + date1 + '_GRAC_HUST_BA01_0600.gfc'
    hust_c3, hust_s3 = LoadCSR().load(fileIn=hust_filename3).getCS(Nmax=60)
    hust_c3, hust_s3 = hust_c3 - static_c, hust_s3 - static_s
    hust_c3, hust_s3 = GeoMathKit.CS_1dTo2d(hust_c3), GeoMathKit.CS_1dTo2d(hust_s3)
    # '''read hust'''
    hust_filename4 = '../result/grace_v3/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + date1 + '_GRAC_HUST_BA01_0600.gfc'
    hust_c4, hust_s4 = LoadCSR().load(fileIn=hust_filename4).getCS(Nmax=60)
    hust_c4, hust_s4 = hust_c4 - static_c, hust_s4 - static_s
    hust_c4, hust_s4 = GeoMathKit.CS_1dTo2d(hust_c4), GeoMathKit.CS_1dTo2d(hust_s4)
    '''read CSR 05'''
    csr_05_name = '../result/SH/CSR 05/CSR_CSR-Release-05_unfiltered_GSM-2_2009001-2009031_0031_UTCSR_0096_0005.gfc'
    csr_05_c, csr_05_s = LoadCSR().load(fileIn=csr_05_name).getCS(Nmax=60)
    csr_05_c, csr_05_s = csr_05_c - static_c, csr_05_s - static_s
    csr_05_c, csr_05_s = GeoMathKit.CS_1dTo2d(csr_05_c), GeoMathKit.CS_1dTo2d(csr_05_s)
    '''read CSR'''
    csr_filename = '../result/SH/CSR/GSM-2_' + date2 + '_GRAC_UTCSR_BA01_0600.gfc'
    csr_c, csr_s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
    csr_c, csr_s = csr_c - static_c, csr_s - static_s
    csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_c), GeoMathKit.CS_1dTo2d(csr_s)
    # csr_diff_c = hust_c - csr_c
    # csr_diff_s = hust_s - csr_s
    # csr_e_l = get_drgee_rms(diff_c=csr_diff_c, diff_s=csr_diff_s)
    # csr_n_l = get_geoid_drgee_error(diff_c=csr_diff_c, diff_s=csr_diff_s)
    '''read GFZ'''
    gfz_filename = '../result/SH/GFZ/GSM-2_' + date2 + '_GRAC_GFZOP_BA01_0600.gfc'
    gfz_c, gfz_s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
    gfz_c, gfz_s = gfz_c - static_c, gfz_s - static_s
    gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)
    '''read JPL'''
    jpl_filename = '../result/SH/JPL/GSM-2_' + date2 + '_GRAC_JPLEM_BA01_0600.gfc'
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
    # static = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    '''csr'''
    csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
    '''csr 05'''
    csr_05 = get_geoid_drgee_error(diff_c=csr_05_c, diff_s=csr_05_s)
    '''gfz'''
    gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
    '''gfz 05'''
    # gfz_05 = get_geoid_drgee_error(diff_c=gfz_05_c, diff_s=gfz_05_s)
    '''jpl'''
    jpl = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)

    matplotlib.rcParams.update({'font.size': 30})
    fig = plt.figure(figsize=(10, 8))
    # plt.style.use(['science'])
    plt.title('%s' % date1, fontsize=30)
    plt.plot(np.arange(2, 61, 1), hust[2:], color='green', linewidth=2.0, marker='', label='Hawk 6h GLS')
    # plt.plot(np.arange(2, 61, 1), hust2[2:], color='black', linewidth=2.0, marker='', label='Hawk 3h OLS')
    # plt.plot(np.arange(2, 61, 1), hust3[2:], color='red', linewidth=2.0, marker='', label='PyHawk-v2')
    # plt.plot(np.arange(2, 61, 1), hust4[2:], color='blue', linewidth=2.0, marker='', label='qr')
    plt.plot(np.arange(2, 61, 1), csr[2:], color='indigo', marker='', linewidth=2.0,label='CSR')
    plt.plot(np.arange(2, 61, 1), gfz[2:], color='#A53860', marker='', linewidth=2.0, label='GFZ')
    plt.plot(np.arange(2, 61, 1), jpl[2:], color='blue', marker='',linewidth=2.0, label='JPL')
    # plt.plot(np.arange(2, 61, 1), csr_05[2:], color='blue', marker='', label='CSR 05')
    # plt.plot(np.arange(2, 61, 1), gfz_05[2:], color='blue', marker='H', label='GFZ 05')
    # plt.plot(np.arange(2, 61, 1), csr_e_l[2:], color='red', marker='s', label='CSR - HUST')
    # plt.plot(np.arange(2, 61, 1), gfz_e_l[2:], color='blue', marker='p', label='GFZ - HUST')
    # plt.plot(np.arange(2, 61, 1), jpl_e_l[2:], color='green', marker='P', label='JPL - HUST')
    # plt.plot(np.arange(2, 61, 1), csr_gfz_e_l[2:], color='black', marker='*', label='GFZ - CSR')
    # plt.plot(np.arange(2, 61, 1), csr_jpl_e_l[2:], color='indigo', marker='h', label='JPL - CSR')
    # plt.plot(np.arange(2, 61, 1), gfz_jpl_e_l[2:], color='tan', marker='H', label='JPL - GFZ')
    plt.xlim(0, 60)
    plt.ylim(4e-5, 1e-2)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('Degree', fontsize=28)
    plt.ylabel('Degree geoid height [m]', fontsize=28)
    plt.yscale("log")
    plt.legend(fontsize=20, loc=1)
    plt.grid(True, which='both', linestyle='-.')
    # plt.savefig('../result/grace_v2/' + date1 + '.png')
    plt.show()
    pass


def GRACE_FO(date):
    date1 = date.replace('_', '-')
    # date2 = date1
    date2 = convert_range(date, standard_to_compact)

    '''static model'''
    static_c, static_s = LoadGOCO06S().load('../data/StaticGravityField/GOCO06s.gfc').getCS(60)
    '''static model'''
    static_goco, static_goco = LoadGOCO06S().load('../data/StaticGravityField/GOCO06s.gfc').getCS(60)
    '''read hust'''

    hust_filename = '../result/case-ESSD/6h/ols/Hawk_60x60_unfiltered_GSM-2_' + date1 + '_GRFO_0100.gfc'
    hust_c, hust_s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
    hust_c, hust_s = hust_c - static_c, hust_s - static_s
    # hust_c, hust_s = hust_c - static_c, hust_s - static_s
    hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
    '''read hust'''
    hust_filename2 = '../result/case-ESSD/6h/gls/Hawk_60x60_unfiltered_GSM-2_' + date1 + '_GRFO_0100.gfc'
    hust_c2, hust_s2 = LoadCSR().load(fileIn=hust_filename2).getCS(Nmax=60)
    hust_c2, hust_s2 = hust_c2 - static_c, hust_s2 - static_s
    hust_c2, hust_s2 = GeoMathKit.CS_1dTo2d(hust_c2), GeoMathKit.CS_1dTo2d(hust_s2)
    '''read hust'''
    hust_filename3 = '../result/case-ESSD/6h/gls/Hawk_60x60_unfiltered_GSM-2_' + date1 + '_GRFO_0100.gfc'
    hust_c3, hust_s3 = LoadCSR().load(fileIn=hust_filename3).getCS(Nmax=60)
    hust_c3, hust_s3 = hust_c3 - static_c, hust_s3 - static_s
    hust_c3, hust_s3 = GeoMathKit.CS_1dTo2d(hust_c3), GeoMathKit.CS_1dTo2d(hust_s3)
    # '''read hust'''
    hust_filename4 = '../result/gracefogls/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + date1 + '_GRAC_HUST_BA01_0600.gfc'
    hust_c4, hust_s4 = LoadCSR().load(fileIn=hust_filename4).getCS(Nmax=60)
    hust_c4, hust_s4 = hust_c4 - static_c, hust_s4 - static_s
    hust_c4, hust_s4 = GeoMathKit.CS_1dTo2d(hust_c4), GeoMathKit.CS_1dTo2d(hust_s4)
    # '''read CSR 05'''
    # csr_05_name = '../result/SH/CSR/GSM-2_2006001-2006031_GRAC_UTCSR_BA01_0600.gfc'
    # csr_05_c, csr_05_s = LoadCSR().load(fileIn=csr_05_name).getCS(Nmax=60)
    # csr_05_c, csr_05_s = csr_05_c - static_c, csr_05_s - static_s
    # csr_05_c, csr_05_s = GeoMathKit.CS_1dTo2d(csr_05_c), GeoMathKit.CS_1dTo2d(csr_05_s)
    '''read CSR'''
    csr_filename = '../result/GFO/GSM-2_' + date2 + '_GRFO_UTCSR_BA01_0603.gfc'
    # csr_filename = '../result/GFO/AIUB_AIUB-GRACE-FO_rl02op_AIUB-GRACE-FO-rl02op_2025-12.gfc'
    csr_c, csr_s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
    csr_c, csr_s = csr_c - static_c, csr_s - static_s
    csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_c), GeoMathKit.CS_1dTo2d(csr_s)
    # csr_diff_c = hust_c - csr_c
    # csr_diff_s = hust_s - csr_s
    # csr_e_l = get_drgee_rms(diff_c=csr_diff_c, diff_s=csr_diff_s)
    # csr_n_l = get_geoid_drgee_error(diff_c=csr_diff_c, diff_s=csr_diff_s)
    '''read GFZ'''
    gfz_filename = '../result/GFO/GSM-2_' + date2 + '_GRFO_GFZOP_BA01_0603.gfc'
    # gfz_filename = '../result/GFO/AIUB_AIUB-GRACE-FO_rl02op_AIUB-GRACE-FO-rl02op_2025-12.gfc'
    gfz_c, gfz_s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
    gfz_c, gfz_s = gfz_c - static_c, gfz_s - static_s
    gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)
    '''read JPL'''
    jpl_filename = '../result/GFO/GSM-2_' + date2 + '_GRFO_JPLEM_BA01_0603.gfc'
    # jpl_filename = '../result/GFO/AIUB_AIUB-GRACE-FO_rl02op_AIUB-GRACE-FO-rl02op_2025-12.gfc'
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
    # static = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    '''csr'''
    csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
    '''csr 05'''
    # csr_05 = get_geoid_drgee_error(diff_c=csr_05_c, diff_s=csr_05_s)
    '''gfz'''
    gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
    '''gfz 05'''
    # gfz_05 = get_geoid_drgee_error(diff_c=gfz_05_c, diff_s=gfz_05_s)
    '''jpl'''
    jpl = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)

    matplotlib.rcParams.update({'font.size': 30})
    fig = plt.figure(figsize=(10, 8))
    # plt.style.use(['science'])
    plt.title('%s' % date, fontsize=30)
    # plt.title('GRACE-FO', fontsize=30)
    plt.plot(np.arange(2, 61, 1), hust[2:], color="#1f77b4", linewidth=2.0, marker='', label='PyHawk OLS')
    plt.plot(np.arange(2, 61, 1), hust2[2:], color='black', linewidth=2.0, marker='', label='PyHawk GLS-8s')
    plt.plot(np.arange(2, 61, 1), hust3[2:], color='red', linewidth=2.0, marker='', label='PyHawk GLS-5s')
    # plt.plot(np.arange(2, 61, 1), hust4[2:], color='blue', linewidth=2.0, marker='', label=_degree_curve_label('PyHawk', title))
    plt.plot(np.arange(2, 61, 1), csr[2:], color='indigo', marker='', linewidth=2.0,label='CSR')
    plt.plot(np.arange(2, 61, 1), gfz[2:], color='#A53860', marker='', linewidth=2.0, label='GFZ')
    plt.plot(np.arange(2, 61, 1), jpl[2:], color='blue', marker='',linewidth=2.0, label='JPL')

    plt.xlim(0, 60)
    plt.ylim(1e-4, 1e-2)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('Degree', fontsize=28)
    plt.ylabel('Degree geoid height [m]', fontsize=28)
    plt.yscale("log")
    plt.legend(fontsize=20, loc=1)
    plt.grid(True, which='both', linestyle='-.')
    # plt.savefig('../result/gracefogls/' + date1 + '.png')
    plt.show()
    pass


def drawTime(data1, data2):
    '''static model'''
    static_c, static_s = LoadGif48().load('../data/StaticGravityField/gif48.gfc').getCS(60)
    '''read hust'''

    hust_filename = '../result/20250728/GRFO/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + data1 + '_GRAC_HUST_BA01_0600-gls.gfc'
    hust_c, hust_s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
    hust_c, hust_s = hust_c - static_c, hust_s - static_s
    # hust_c, hust_s = hust_c - static_c, hust_s - static_s
    hust_c, hust_s = GeoMathKit.CS_1dTo2d(hust_c), GeoMathKit.CS_1dTo2d(hust_s)
    # '''read hust'''
    hust_filename2 = '../result/20250728/GRFO/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + data1 + '_GRAC_HUST_BA01_0600.gfc'
    hust_c2, hust_s2 = LoadCSR().load(fileIn=hust_filename2).getCS(Nmax=60)
    hust_c2, hust_s2 = hust_c2 - static_c, hust_s2 - static_s
    hust_c2, hust_s2 = GeoMathKit.CS_1dTo2d(hust_c2), GeoMathKit.CS_1dTo2d(hust_s2)
    '''read hust'''
    hust_filename3 = '../result/20250728/GRFO/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + data1 + '_GRAC_HUST_BA01_0600.gfc'
    hust_c3, hust_s3 = LoadCSR().load(fileIn=hust_filename3).getCS(Nmax=60)
    hust_c3, hust_s3 = hust_c3 - static_c, hust_s3 - static_s
    hust_c3, hust_s3 = GeoMathKit.CS_1dTo2d(hust_c3), GeoMathKit.CS_1dTo2d(hust_s3)
    # '''read hust'''
    hust_filename4 = '../result/20250728/GRFO/HUST_HUST-Release-06_60x60_unfiltered_GSM-2_' + data1 + '_GRAC_HUST_BA01_0600.gfc'
    hust_c4, hust_s4 = LoadCSR().load(fileIn=hust_filename4).getCS(Nmax=60)
    hust_c4, hust_s4 = hust_c4 - static_c, hust_s4 - static_s
    hust_c4, hust_s4 = GeoMathKit.CS_1dTo2d(hust_c4), GeoMathKit.CS_1dTo2d(hust_s4)
    # '''read CSR 05'''
    # csr_05_name = '../result/SH/CSR/GSM-2_2006001-2006031_GRAC_UTCSR_BA01_0600.gfc'
    # csr_05_c, csr_05_s = LoadCSR().load(fileIn=csr_05_name).getCS(Nmax=60)
    # csr_05_c, csr_05_s = csr_05_c - static_c, csr_05_s - static_s
    # csr_05_c, csr_05_s = GeoMathKit.CS_1dTo2d(csr_05_c), GeoMathKit.CS_1dTo2d(csr_05_s)
    '''read CSR'''
    # csr_filename = '../result/SH/CSR/GSM-2_' + data2 + '_GRAC_UTCSR_BA01_0600.gfc'
    csr_filename = '../result/GFO/GSM-2_' + data2 + '_GRFO_UTCSR_BA01_0600.gfc'
    csr_c, csr_s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
    csr_c, csr_s = csr_c - static_c, csr_s - static_s
    csr_c, csr_s = GeoMathKit.CS_1dTo2d(csr_c), GeoMathKit.CS_1dTo2d(csr_s)
    # csr_diff_c = hust_c - csr_c
    # csr_diff_s = hust_s - csr_s
    # csr_e_l = get_drgee_rms(diff_c=csr_diff_c, diff_s=csr_diff_s)
    # csr_n_l = get_geoid_drgee_error(diff_c=csr_diff_c, diff_s=csr_diff_s)
    '''read GFZ'''
    # gfz_filename = '../result/SH/GFZ/GSM-2_' + data2 + '_GRAC_GFZOP_BA01_0600.gfc'
    gfz_filename = '../result/GFO/GSM-2_' + data2 + '_GRFO_GFZOP_BA01_0600.gfc'
    gfz_c, gfz_s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
    gfz_c, gfz_s = gfz_c - static_c, gfz_s - static_s
    gfz_c, gfz_s = GeoMathKit.CS_1dTo2d(gfz_c), GeoMathKit.CS_1dTo2d(gfz_s)
    # '''read GFZ 05'''
    # gfz_05_name = '../compare/GFZ_GFZ-Release-05_monthly_unfiltered_GSM-2_2008001-2008031_0031_EIGEN_G---_005a.gfc'
    # gfz_05_c, gfz_05_s = LoadCSR().load(fileIn=gfz_05_name).getCS(Nmax=60)
    # gfz_05_c, gfz_05_s = gfz_05_c - static_c, gfz_05_s - static_s
    # gfz_05_c, gfz_05_s = GeoMathKit.CS_1dTo2d(gfz_05_c), GeoMathKit.CS_1dTo2d(gfz_05_s)
    # # gfz_diff_c = hust_c - gfz_c
    # # gfz_diff_s = hust_s - gfz_s
    # # gfz_e_l = get_drgee_rms(diff_c=gfz_diff_c, diff_s=gfz_diff_s)
    # # gfz_n_l = get_geoid_drgee_error(diff_c=gfz_diff_c, diff_s=gfz_diff_s)
    '''read JPL'''
    # jpl_filename = '../result/SH/JPL/GSM-2_' + data2 + '_GRAC_JPLEM_BA01_0600.gfc'
    jpl_filename = '../result/GFO/GSM-2_' + data2 + '_GRFO_JPLEM_BA01_0600.gfc'
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
    # static = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    '''csr'''
    csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
    '''csr 05'''
    # csr_05 = get_geoid_drgee_error(diff_c=csr_05_c, diff_s=csr_05_s)
    '''gfz'''
    gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
    '''gfz 05'''
    # gfz_05 = get_geoid_drgee_error(diff_c=gfz_05_c, diff_s=gfz_05_s)
    '''jpl'''
    jpl = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)
    # plt.title('Geoid Degree', fontsize=30)
    matplotlib.rcParams.update({'font.size': 30})
    fig = plt.figure(figsize=(10, 8))
    # plt.style.use(['science'])
    plt.plot(np.arange(2, 61, 1), hust[2:], color='green', linewidth=2.0, marker='', label='limit')
    # plt.plot(np.arange(2, 61, 1), hust2[2:], color='black', linewidth=2.0, marker='', label='non')
    # plt.plot(np.arange(2, 61, 1), hust3[2:], color='red', linewidth=2.0, marker='', label='6hacc')
    # plt.plot(np.arange(2, 61, 1), hust4[2:], color='blue', linewidth=2.0, marker='', label='0.003')
    # plt.plot(np.arange(2, 61, 1), csr[2:], color='indigo', marker='', linewidth=2.0,label=_degree_curve_label('CSR', title))
    plt.plot(np.arange(2, 61, 1), gfz[2:], color='#A53860', marker='', linewidth=2.0, label=_degree_curve_label('GFZ', title))
    # plt.plot(np.arange(2, 61, 1), jpl[2:], color='blue', marker='',linewidth=2.0, label=_degree_curve_label('JPL', title))
    # plt.plot(np.arange(2, 61, 1), csr_05[2:], color='indigo', marker='h', label='CSR 05')
    # plt.plot(np.arange(2, 61, 1), gfz_05[2:], color='blue', marker='H', label='GFZ 05')
    # plt.plot(np.arange(2, 61, 1), csr_e_l[2:], color='red', marker='s', label='CSR - HUST')
    # plt.plot(np.arange(2, 61, 1), gfz_e_l[2:], color='blue', marker='p', label='GFZ - HUST')
    # plt.plot(np.arange(2, 61, 1), jpl_e_l[2:], color='green', marker='P', label='JPL - HUST')
    # plt.plot(np.arange(2, 61, 1), csr_gfz_e_l[2:], color='black', marker='*', label='GFZ - CSR')
    # plt.plot(np.arange(2, 61, 1), csr_jpl_e_l[2:], color='indigo', marker='h', label='JPL - CSR')
    # plt.plot(np.arange(2, 61, 1), gfz_jpl_e_l[2:], color='tan', marker='H', label='JPL - GFZ')
    plt.xlim(0, 60)
    plt.ylim(1e-4, 1e-2)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('Degree', fontsize=28)
    plt.ylabel('Degree geoid height [m]', fontsize=28)
    plt.yscale("log")
    plt.legend(fontsize=20, loc=1)
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


def drawMean(year):
    '''static model'''
    static_c, static_s = LoadGOCO06S().load('../data/StaticGravityField/GOCO06s.gfc').getCS(60)
    '''read hust'''
    filenamelist = os.listdir(r'../result/hawk_rl01/gracefo/'+year+'/hawk')
    hust_c, hust_s = 0, 0
    i = 0
    degree_sum = np.zeros(61)
    for filename in filenamelist:
        hust_filename = '../result/hawk_rl01/gracefo/'+year+'/hawk/'+ filename
        c, s = LoadCSR().load(fileIn=hust_filename).getCS(Nmax=60)
        c -= static_c
        s -= static_s

        c, s = GeoMathKit.CS_1dTo2d(c), GeoMathKit.CS_1dTo2d(s)

        degree_curve = get_geoid_drgee_error(diff_c=c, diff_s=s)
        degree_sum += degree_curve

        i += 1

    hust = degree_sum / i

    '''read CSR'''
    filenamelist = os.listdir(r'../result/hawk_rl01/gracefo/'+year+'/CSR')
    csr_c, csr_s = 0, 0
    j = 0
    degree_sum = np.zeros(61)
    for filename in filenamelist:
        csr_filename = '../result/hawk_rl01/gracefo/'+year+'/csr/'+ filename
        c, s = LoadCSR().load(fileIn=csr_filename).getCS(Nmax=60)
        c, s = c - static_c, s - static_s
        c, s = GeoMathKit.CS_1dTo2d(c), GeoMathKit.CS_1dTo2d(s)

        degree_curve = get_geoid_drgee_error(diff_c=c, diff_s=s)
        degree_sum += degree_curve
        j += 1

    csr = degree_sum / j

    '''read GFZ'''
    filenamelist = os.listdir(r'../result/hawk_rl01/gracefo/'+year+'/GFZ')
    gfz_c, gfz_s = 0, 0
    k = 0
    degree_sum = np.zeros(61)
    for filename in filenamelist:
        gfz_filename = '../result/hawk_rl01/gracefo/'+year+'/GFZ/' + filename
        c, s = LoadCSR().load(fileIn=gfz_filename).getCS(Nmax=60)
        c, s = c - static_c, s - static_s
        c, s = GeoMathKit.CS_1dTo2d(c), GeoMathKit.CS_1dTo2d(s)
        degree_curve = get_geoid_drgee_error(diff_c=c, diff_s=s)
        degree_sum += degree_curve

        k += 1

    gfz = degree_sum / k
    '''read JPL'''
    filenamelist = os.listdir(r'../result/hawk_rl01/gracefo/'+year+'/JPL')
    jpl_c, jpl_s = 0, 0
    q = 0
    degree_sum = np.zeros(61)
    for filename in filenamelist:
        jpl_filename = '../result/hawk_rl01/gracefo/'+year+'/JPL/' + filename
        jpl_c, jpl_s = LoadCSR().load(fileIn=jpl_filename).getCS(Nmax=60)
        jpl_c, jpl_s = jpl_c - static_c, jpl_s - static_s
        jpl_c, jpl_s = GeoMathKit.CS_1dTo2d(jpl_c), GeoMathKit.CS_1dTo2d(jpl_s)

        degree_curve = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)
        degree_sum += degree_curve
        q += 1

    jpl = degree_sum / q
    '''hust'''
    # hust = get_geoid_drgee_error(diff_c=hust_c, diff_s=hust_s)
    '''csr'''
    # csr = get_geoid_drgee_error(diff_c=csr_c, diff_s=csr_s)
    '''gfz'''
    # gfz = get_geoid_drgee_error(diff_c=gfz_c, diff_s=gfz_s)
    '''jpl'''
    # jpl = get_geoid_drgee_error(diff_c=jpl_c, diff_s=jpl_s)
    plt.title('GRACE-FO(' + year + ')', fontsize=30)
    # matplotlib.rcParams.update({'font.size': 30})
    # plt.style.use(['science'])

    plt.plot(np.arange(2, 61, 1), hust[2:], color='green', linewidth=3.0, marker='', label='Hawk RL01')
    plt.plot(np.arange(2, 61, 1), gfz[2:], color='black', marker='', linewidth=3.0, label='GFZ RL06')
    plt.plot(np.arange(2, 61, 1), jpl[2:], color='blue', marker='', linewidth=3.0, label='JPL RL06')
    plt.plot(np.arange(2, 61, 1), csr[2:], color='red', marker='', linewidth=3.0, label='CSR RL06')
    plt.xlim(0, 60)
    plt.ylim(1e-5, 1e-1)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('Degree', fontsize=28)
    plt.ylabel('Degree geoid height [m]', fontsize=28)
    plt.yscale("log")
    plt.legend(fontsize=20, loc=1)
    plt.grid(True, which='both', linestyle='-.')
    plt.show()


def standard_to_compact(date_str):
    """标准格式 → 紧凑格式 (2021-01-01 → 2021001)"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = date_obj.timetuple().tm_yday  # 获取年内第几天
    return f"{date_obj.year}{day_of_year:03d}"  # 年份+三位数日


def convert_range(date_range, converter):
    start, end = date_range.split("_")
    return f"{converter(start)}-{converter(end)}"


CLASSIFIED_EXCLUDE_YEAR_MONTHS = {
    (2020, 1),
}


def _compact_to_standard(date_str):
    return datetime.strptime(date_str, "%Y%j").strftime("%Y-%m-%d")


def _classified_date_range(file_in):
    name = Path(file_in).name
    match = re.search(r'GSM-2_(\d{4}-\d{2}-\d{2})[-_](\d{4}-\d{2}-\d{2})_', name)
    if match is not None:
        return f"{match.group(1)}_{match.group(2)}"

    match = re.search(r'GSM-2_(\d{7})-(\d{7})_', name)
    if match is not None:
        return f"{_compact_to_standard(match.group(1))}_{_compact_to_standard(match.group(2))}"

    return None


def _classified_static_model(mission, nmax=60):
    if mission == 'gracefo':
        static_file = BASE_DIR / 'data' / 'StaticGravityField' / 'GOCO06s.gfc'
        return LoadGOCO06S().load(str(static_file)).getCS(nmax)
    if mission == 'grace':
        static_file = BASE_DIR / 'data' / 'StaticGravityField' / 'gif48.gfc'
        return LoadGif48().load(str(static_file)).getCS(nmax)
    raise ValueError(f"Unsupported mission: {mission}")


def _classified_root(mission):
    if mission not in ('grace', 'gracefo'):
        raise ValueError(f"Unsupported mission: {mission}")
    return BASE_DIR / 'result' / 'hawk_rl01' / mission


def _classified_year_month(date_range):
    start = datetime.strptime(date_range.split('_')[0], "%Y-%m-%d")
    return start.year, start.month


def _classified_is_excluded(date_range, exclude_year_months=None):
    exclude_year_months = CLASSIFIED_EXCLUDE_YEAR_MONTHS if exclude_year_months is None else set(exclude_year_months)
    return _classified_year_month(date_range) in exclude_year_months


def _classified_file_index(year_dir, center):
    folder = Path(year_dir) / center
    index = {}
    if not folder.exists():
        return index
    for file_in in sorted(folder.glob('*.gfc')):
        date_range = _classified_date_range(file_in)
        if date_range is not None:
            index[date_range] = file_in
    return index


def _classified_average_coefficients(files, nmax=60):
    c_sum, s_sum = None, None
    count = 0
    for file_in in files:
        c, s = LoadCSR().load(fileIn=str(file_in)).getCS(Nmax=nmax)
        if c_sum is None:
            c_sum = np.zeros_like(c)
            s_sum = np.zeros_like(s)
        c_sum += c
        s_sum += s
        count += 1
    if count == 0:
        raise ValueError("No files were selected for annual mean")
    return c_sum / count, s_sum / count


def _classified_curve_labels(mission):
    release = 'RL06.3' if mission == 'gracefo' else 'RL06'
    return {
        'hawk': 'Hawk RL01',
        'gfz': f'GFZ {release}',
        'jpl': f'JPL {release}',
        'csr': f'CSR {release}',
    }


def _plot_classified_annual_curves(degree_curves, mission, year, out_file, show=False):
    labels = _classified_curve_labels(mission)
    mission_label = 'GRACE-FO' if mission == 'gracefo' else 'GRACE'
    fig = plt.figure(figsize=(10, 8))
    plt.title(f'{mission_label} ({year})', fontsize=30)
    plt.plot(np.arange(2, 61, 1), degree_curves['hawk'][2:], color='green', linewidth=3.0, marker='', label=labels['hawk'])
    plt.plot(np.arange(2, 61, 1), degree_curves['gfz'][2:], color='black', marker='', linewidth=3.0, label=labels['gfz'])
    plt.plot(np.arange(2, 61, 1), degree_curves['jpl'][2:], color='blue', marker='', linewidth=3.0, label=labels['jpl'])
    plt.plot(np.arange(2, 61, 1), degree_curves['csr'][2:], color='red', marker='', linewidth=3.0, label=labels['csr'])
    plt.xlim(0, 60)
    plt.ylim(1e-5, 1e-1)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel('Degree', fontsize=28)
    plt.ylabel('Degree geoid height [m]', fontsize=28)
    plt.yscale("log")
    plt.legend(fontsize=20, loc=1)
    plt.grid(True, which='both', linestyle='-.')
    fig.tight_layout()
    plt.savefig(out_file, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def draw_classified_annual_mean(mission, year, out_dir=None, exclude_year_months=None, nmax=60):
    root = _classified_root(mission)
    year_dir = root / str(year)
    out_dir = Path(out_dir) if out_dir is not None else root
    out_dir.mkdir(parents=True, exist_ok=True)

    indexes = {center: _classified_file_index(year_dir, center) for center in ('hawk', 'csr', 'gfz', 'jpl')}
    common_ranges = set(indexes['hawk'])
    for center in ('csr', 'gfz', 'jpl'):
        common_ranges &= set(indexes[center])
    used_ranges = [date_range for date_range in sorted(common_ranges)
                   if not _classified_is_excluded(date_range, exclude_year_months)]
    if not used_ranges:
        raise ValueError(f"No common months for {mission} {year}")

    static_c, static_s = _classified_static_model(mission, nmax=nmax)
    degree_curves = {}
    for center in ('hawk', 'csr', 'gfz', 'jpl'):
        degree_sum = np.zeros(nmax + 1)
        for date_range in used_ranges:
            file_in = indexes[center][date_range]
            c, s = LoadCSR().load(fileIn=str(file_in)).getCS(Nmax=nmax)
            c, s = c - static_c, s - static_s
            c_2d, s_2d = GeoMathKit.CS_1dTo2d(c), GeoMathKit.CS_1dTo2d(s)
            degree_sum += get_geoid_drgee_error(diff_c=c_2d, diff_s=s_2d)
        degree_curves[center] = degree_sum / len(used_ranges)

    stem = f'annual_mean_degree_{mission}_{year}'
    png_file = out_dir / f'{stem}.png'
    csv_file = out_dir / f'{stem}.csv'
    npz_file = out_dir / f'annual_mean_field_{mission}_{year}.npz'

    _plot_classified_annual_curves(degree_curves, mission, year, png_file, show=False)
    rows = np.column_stack((
        np.arange(0, nmax + 1, 1),
        degree_curves['hawk'],
        degree_curves['csr'],
        degree_curves['gfz'],
        degree_curves['jpl'],
    ))
    np.savetxt(csv_file, rows, delimiter=',', header='degree,Hawk,CSR,GFZ,JPL', comments='')
    np.savez(
        npz_file,
        date_ranges=np.array(used_ranges, dtype='U32'),
        month_count=len(used_ranges),
        degree=np.arange(0, nmax + 1, 1),
        Hawk_degree=degree_curves['hawk'],
        CSR_degree=degree_curves['csr'],
        GFZ_degree=degree_curves['gfz'],
        JPL_degree=degree_curves['jpl'],
    )
    print(f"[{mission} {year}] used {len(used_ranges)} month(s)")
    return {'mission': mission, 'year': year, 'used': used_ranges, 'png': png_file, 'csv': csv_file, 'npz': npz_file}


def draw_classified_annual_4year(mission, years, out_dir=None, nmax=60):
    root = _classified_root(mission)
    out_dir = Path(out_dir) if out_dir is not None else root
    labels = _classified_curve_labels(mission)
    mission_label = 'GRACE-FO' if mission == 'gracefo' else 'GRACE'
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, year in zip(axes, years):
        csv_file = out_dir / f'annual_mean_degree_{mission}_{year}.csv'
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        degree = data[:, 0]
        curves = {
            'hawk': data[:, 1],
            'csr': data[:, 2],
            'gfz': data[:, 3],
            'jpl': data[:, 4],
        }
        ax.plot(degree[2:], curves['hawk'][2:], color='green', linewidth=3.0, marker='', label=labels['hawk'])
        ax.plot(degree[2:], curves['gfz'][2:], color='black', marker='', linewidth=3.0, label=labels['gfz'])
        ax.plot(degree[2:], curves['jpl'][2:], color='blue', marker='', linewidth=3.0, label=labels['jpl'])
        ax.plot(degree[2:], curves['csr'][2:], color='red', linewidth=3.0, marker='', label=labels['csr'])
        ax.set_title(f'{mission_label} ({year})', fontsize=30)
        ax.set_xlim(0, nmax)
        ax.set_ylim(1e-5, 1e-1)
        ax.set_xlabel('Degree', fontsize=28)
        ax.set_ylabel('Degree geoid height [m]', fontsize=28)
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=28)
        ax.legend(fontsize=20, loc=1)
        ax.grid(True, which='both', linestyle='-.')
    for ax in axes[len(years):]:
        ax.axis('off')
    fig.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.5)
    out_file = out_dir / f'annual_mean_degree_{mission}_{years[0]}_{years[-1]}_4year_merged.png'
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"wrote {out_file}")
    return out_file


def batch_classified_annual_means():
    outputs = []
    for mission in ('grace', 'gracefo'):
        root = _classified_root(mission)
        years = sorted(int(folder.name) for folder in root.iterdir()
                       if folder.is_dir() and folder.name.isdigit())
        mission_outputs = []
        used_years = []
        for year in years:
            try:
                item = draw_classified_annual_mean(mission, year, out_dir=root)
            except ValueError as exc:
                print(f"[{mission} {year}] skipped: {exc}")
                continue
            mission_outputs.append(item)
            used_years.append(year)
        for i in range(0, len(used_years), 4):
            draw_classified_annual_4year(mission, used_years[i:i + 4], out_dir=root)
        outputs.extend(mission_outputs)

        audit_lines = []
        for item in mission_outputs:
            audit_lines.append(f"{item['year']}: month_count={len(item['used'])}; used={','.join(item['used'])}")
        (root / 'annual_mean_audit.txt').write_text('\n'.join(audit_lines), encoding='utf-8')
    return outputs


if __name__ == '__main__':

    span1 = '2020-03-01_2020-03-31'
    # GRACE(span1)
    GRACE_FO(span1)

    # drawMean('2019')
    pass










