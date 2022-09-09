from src.ForceModel.ConfigGravHar import ConfigGravHar
from src.ForceModel.OceanTide import EOT11a, FES2014
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat,GravHarType
from src.ForceModel.BaseGravCS import BaseGravCS
from src.ForceModel.GravHar import ClibGravHar
import numpy as np


def demo1():
    config_model = ConfigGravHar()
    fr = Frame(EOP(True).load())

    time = fr.setTime(505051205, TimeFormat.GPS_second)
    config_model = config_model.setTime(time)
    print(config_model.getSumCS())


if __name__ == '__main__':
    demo1()

