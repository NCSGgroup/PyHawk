from src.ForceModel.OceanTide import EOT11a, FES2014
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat
from src.ForceModel.BaseForceCS import BaseForceCS


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(252417548.816000, TimeFormat.GPS_second)

    eot = EOT11a()
    eot.setTime(fr).load(90).getCS()
    eot.accumulate()

    fes = FES2014()
    fes.setTime(fr).load(90).getCS()


if __name__ == '__main__':
    demo1()

