# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:54
# @File    : demo_BenchMark.py
# @Software: PyCharm
from demo.BenchMarkForJson import BenchMark
from src.Preference.ForceModelConfig import ForceModelConfig
import json


def demo1():
    Obj2 = ForceModelConfig()
    dict2 = json.load(open('../setting/BenchMarkTest_ForceModel.json', 'r'))
    Obj2.__dict__ = dict2
    benchMark = BenchMark(Obj2)
    pass


if __name__ == '__main__':
    demo1()