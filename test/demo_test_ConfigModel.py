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
    fr.setTime(505051205, TimeFormat.GPS_second)
    pos = np.array([-4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06])

    cm = config_model.setTime(fr)
    c,s = cm.getSumCS()
    gh = ClibGravHar().setConstant().setPar(c, s, pos)
    print(gh.getAcceleration())


if __name__ == '__main__':
    demo1()