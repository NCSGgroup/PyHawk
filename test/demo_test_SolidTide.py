from src.ForceModel.SolidTide import SolidTideIERS2010
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat
from src.Frame.PlanetEphemerides import PlanetEphemerides


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(51544.5, TimeFormat.GPS_second)

    ep = PlanetEphemerides().configure().setPlanets().setTime(fr)

    c, s = SolidTideIERS2010(ep).setTime(fr).getCS()
    print(c)
    print(s)


if __name__ == '__main__':
    demo1()