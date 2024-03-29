# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:54
# @File    : demo_BenchMark.py
# @Software: PyCharm
from benchmark.BenchMarkForJson import BenchMark
from src.Preference.Pre_ForceModel import ForceModelConfig
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import json


def demo1():
    FMConfig = ForceModelConfig()
    benchMark = BenchMark()
    '''three body'''
    threeBody = json.load(open('../setting/BenchMarkConfig/BenchMark_ThreeBody.json', 'r'))
    FMConfig.__dict__ = threeBody
    benchMark.three_body(FMConfig)
    '''AOD'''
    aod = json.load(open('../setting/BenchMarkConfig/BenchMark_AOD.json', 'r'))
    FMConfig.__dict__ = aod
    benchMark.aod(FMConfig)
    '''relativistic'''
    relativistic = json.load(open('../setting/BenchMarkConfig/BenchMark_Relativity.json', 'r'))
    FMConfig.__dict__ = relativistic
    benchMark.relativistic(FMConfig)
    '''eot11a'''
    eot11a = json.load(open('../setting/BenchMarkConfig/BenchMark_EOT11a.json', 'r'))
    FMConfig.__dict__ = eot11a
    benchMark.eot11a(FMConfig)
    '''FES2014'''
    fes2014 = json.load(open('../setting/BenchMarkConfig/BenchMark_FES2104.json', 'r'))
    FMConfig.__dict__ = fes2014
    benchMark.fes2014b(FMConfig)
    '''Atmos tide'''
    atmos = json.load(open('../setting/BenchMarkConfig/BenchMark_Atmos.json', 'r'))
    FMConfig.__dict__ = atmos
    benchMark.atmos_tide(FMConfig)
    '''gravity field'''
    gravity_field = json.load(open('../setting/BenchMarkConfig/BenchMark_Gravity.json', 'r'))
    FMConfig.__dict__ = gravity_field
    benchMark.gravity_field(FMConfig)
    '''solidEarth tide'''
    solid_earth = json.load(open('../setting/BenchMarkConfig/BenchMark_SolidEarth.json', 'r'))
    FMConfig.__dict__ = solid_earth
    benchMark.solid_earth(FMConfig)
    '''solidEarth pole tide'''
    earth_pole = json.load(open('../setting/BenchMarkConfig/BenchMark_EarthPole.json', 'r'))
    FMConfig.__dict__ = earth_pole
    benchMark.earth_pole(FMConfig)
    '''ocean pole tide'''
    ocean_pole = json.load(open('../setting/BenchMarkConfig/BenchMark_OceanPole.json', 'r'))
    FMConfig.__dict__ = ocean_pole
    benchMark.ocean_pole(FMConfig)
    pass


def demo2():
    diff = np.load('../result/benchmarktest/sun.npy')
    timeList = np.shape(diff)[0]
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    rcParams['font.size'] = 24  # 设置字体大小为14
    rcParams['font.family'] = 'serif'  # 设置字体为衬线字体

    plt.title('Sun', fontsize=24)
    plt.plot(ax, color='#dd2c00', label='x')
    plt.plot(ay, color='#00695c', label='y')
    plt.plot(az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD', fontsize=24)
    plt.ylabel(r'$m/s^2$', fontsize=24)
    # plt.savefig('../result/benchmarktest/AODRL06.png')
    plt.show()


if __name__ == '__main__':
    # demo1()
    demo2()