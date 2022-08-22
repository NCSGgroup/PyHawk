from src.ForceModel.OceanTide import EOT11a, FES2014
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat
from src.ForceModel.BaseGravCS import BaseGravCS
from src.ForceModel.GravHar import ClibGravHar


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(252417548.816000, TimeFormat.GPS_second)

    eot = EOT11a()
    c, s = eot.setTime(fr).load(100).getCS()

    eot2 = EOT11a()
    c2, s2 = eot2.setTime(fr).load(120).getCS()

    baseCS = BaseGravCS()
    baseCS.accumulate(c, s).accumulate(c2, s2)
    print(c2[0:51*101]-c)


def demo2():
    fr = Frame(EOP(True).load())
    fr.setTime(252417548.816000, TimeFormat.GPS_second)

    eot = EOT11a()
    c, s = eot.setTime(fr).load(100).getCS()

    eot2 = EOT11a()
    c2, s2 = eot2.setTime(fr).load(120).getCS()

    baseCS = BaseGravCS()
    baseCS.accumulate(c, s).accumulate(c2, s2)
    print(baseCS.getTotalCS())

    cgh = ClibGravHar().setConstant().configure(c, s, 555220)


if __name__ == '__main__':
    demo1()

