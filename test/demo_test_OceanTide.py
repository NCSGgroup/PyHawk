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

    print(c, s)


def demo2():
    fr = Frame(EOP(True).load())
    fr.setTime(252417548.816000, TimeFormat.GPS_second)

    eot = EOT11a()
    c, s = eot.setTime(fr).load(100).getCS()

    eot2 = EOT11a()
    c2, s2 = eot2.setTime(fr).load(120).getCS()

    baseCS = BaseGravCS()

    cgh = ClibGravHar().setConstant().configure(c, s, 555220)


def createInstance(module_name, class_name, *args, **kwargs):
    module_meta = __import__(module_name, globals(), locals(), [class_name])
    class_meta = getattr(module_meta, class_name)
    obj = class_meta(*args, **kwargs)
    return obj


if __name__ == '__main__':
    # obj = createInstance("src.ForceModel.OceanTide", "EOT11a")
    # fr = Frame(EOP(True).load())
    # fr.setTime(252417548.816000, TimeFormat.GPS_second)
    #
    # at = obj.setTime(fr).load(100)
    # print(at.getCS())
    demo1()

