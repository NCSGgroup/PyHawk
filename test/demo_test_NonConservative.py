from src.ForceModel.NonConservative import NonConByAcc
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(252417605.816000, TimeFormat.GPS_second)

    NC = NonConByAcc().configure().setTime(fr)
    print(NC.getAcceleration())


if __name__ == '__main__':
    demo1()
