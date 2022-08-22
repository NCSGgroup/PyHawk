from src.ForceModel.PoleTide import PoleTideIERS2010
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(51544.5, TimeFormat.GPS_second)

    po = PoleTideIERS2010().load().setTime(fr)
    c,s = po.getCS()
    print(c,s)


if __name__ == '__main__':
    demo1()

