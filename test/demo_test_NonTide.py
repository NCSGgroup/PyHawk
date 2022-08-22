from src.ForceModel.NonTide import AODRL06
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(32529600.500000, TimeFormat.GPS_second)

    aod = AODRL06().configure().setTime(fr)
    print(aod.getCS())


if __name__ == '__main__':
    demo1()