from src.ForceModel.ConfigGravHar import ConfigGravHar
from src.ForceModel.OceanTide import EOT11a, FES2014
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat,GravHarType
from src.ForceModel.BaseGravCS import BaseGravCS
from src.ForceModel.GravHar import ClibGravHar


def demo1():
    config_model = ConfigGravHar().configure()
    fr = Frame(EOP(True).load())
    fr.setTime(252417548.816000, TimeFormat.GPS_second)

    eot = EOT11a().setType(GravHarType.OceanTide).setTime(fr).load(100)
    eot2 = EOT11a().setType(GravHarType.OceanTide).setTime(fr).load(120)

    config_model.accumulate(eot).accumulate(eot2)
    c, s = config_model.getSumCS()
    # config_model.accumulate(eot2)

    pass
    # eot2 = EOT11a()
    # c2, s2 = eot2.setTime(fr).load(120).getCS()


if __name__ == '__main__':
    demo1()