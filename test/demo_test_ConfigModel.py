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
    print(config_model.GravHarForce)


if __name__ == '__main__':
    demo1()

