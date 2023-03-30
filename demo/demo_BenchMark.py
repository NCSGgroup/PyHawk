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
    '''relativistic'''
    aod = json.load(open('../setting/BenchMarkConfig/BenchMark_Relativity.json', 'r'))
    FMConfig.__dict__ = aod
    benchMark.relativistic(FMConfig)
    pass


if __name__ == '__main__':
    demo1()