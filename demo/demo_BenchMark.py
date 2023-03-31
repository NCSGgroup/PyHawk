# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:54
# @File    : demo_BenchMark.py
# @Software: PyCharm
from demo.BenchMarkForJson import BenchMark
from src.Preference.ForceModelConfig import ForceModelConfig
import json


def demo1():
    FMConfig = ForceModelConfig()
    benchMark = BenchMark()
    # '''three body'''
    # threeBody = json.load(open('../setting/BenchMarkConfig/BenchMark_ThreeBody.json', 'r'))
    # FMConfig.__dict__ = threeBody
    # benchMark.three_body(FMConfig)
    # '''AOD'''
    # aod = json.load(open('../setting/BenchMarkConfig/BenchMark_AOD.json', 'r'))
    # FMConfig.__dict__ = aod
    # benchMark.aod(FMConfig)
    # '''relativistic'''
    # relativistic = json.load(open('../setting/BenchMarkConfig/BenchMark_Relativity.json', 'r'))
    # FMConfig.__dict__ = relativistic
    # benchMark.relativistic(FMConfig)
    # '''eot11a'''
    # eot11a = json.load(open('../setting/BenchMarkConfig/BenchMark_EOT11a.json', 'r'))
    # FMConfig.__dict__ = eot11a
    # benchMark.eot11a(FMConfig)
    # '''FES2014'''
    # fes2014 = json.load(open('../setting/BenchMarkConfig/BenchMark_FES2104.json', 'r'))
    # FMConfig.__dict__ = fes2014
    # benchMark.fes2014b(FMConfig)
    # '''Atmos tide'''
    # atmos = json.load(open('../setting/BenchMarkConfig/BenchMark_Atmos.json', 'r'))
    # FMConfig.__dict__ = atmos
    # benchMark.atmos_tide(FMConfig)
    # '''gravity field'''
    # gravity_field = json.load(open('../setting/BenchMarkConfig/BenchMark_Gravity.json', 'r'))
    # FMConfig.__dict__ = gravity_field
    # benchMark.gravity_field(FMConfig)
    # '''solidEarth tide'''
    # solid_earth = json.load(open('../setting/BenchMarkConfig/BenchMark_SolidEarth.json', 'r'))
    # FMConfig.__dict__ = solid_earth
    # benchMark.solid_earth(FMConfig)
    '''solidEarth pole tide'''
    earth_pole = json.load(open('../setting/BenchMarkConfig/BenchMark_EarthPole.json', 'r'))
    FMConfig.__dict__ = earth_pole
    benchMark.earth_pole(FMConfig)
    # '''socean pole tide'''
    # earth_pole = json.load(open('../setting/BenchMarkConfig/BenchMark_EarthPole.json', 'r'))
    # FMConfig.__dict__ = earth_pole
    # benchMark.earth_pole(FMConfig)
    pass


if __name__ == '__main__':
    demo1()