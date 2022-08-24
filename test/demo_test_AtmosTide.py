from src.ForceModel.AtmosTide import AtmosBB2003, AtmosAOD
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(51544.5, TimeFormat.GPS_second)

    at = AtmosBB2003().setTime(fr)
    C0, S0 = at.getCS()

    print(at.getCS())


def demo2():
    fr = Frame(EOP(True).load())
    fr.setTime(51544.5, TimeFormat.GPS_second)

    at = AtmosAOD().setTime(fr).loadfile()
    print(at.getCS())


if __name__ == '__main__':
    demo1()