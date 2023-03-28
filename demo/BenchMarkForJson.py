# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:48
# @File    : BenchMarkForJson.py
# @Software: PyCharm

from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.ForceModelConfig import ForceModelConfig
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
import src.Preference.EnumType as EnumType
import numpy as np
import matplotlib.pyplot as plt


class BenchMark:
    def __init__(self, FMConfig = ForceModelConfig()):
        self._fr = Frame(EOP().load())
        self._assembleModel = Assemble2ndDerivative().configure(FMConfig=FMConfig)

        pass
