from src.ForceModel.ThreeBody import ThreeBody, ThreeBodyPlusJ2
from src.ForceModel.PlanetEphemerides import PlanetEphemerides
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(55682.000000000000, TimeFormat.UTC_MJD)

    ep = PlanetEphemerides().configure().setPlanets()
    planets = ep.getPosByFrame(fr)
    # print(planets)

    ap = ThreeBodyPlusJ2().setTime(fr)
    ap.setPlanetPos(planets)
    ap.setSatPos([6378136 + 1000000., 6378136 + 2000000., 6378136 + 3000000.])

    acc = ap.get_indirectJ2()
    print(acc)
    # print(du2xyz)


if __name__ == '__main__':
    demo1()

