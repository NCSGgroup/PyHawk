from src.ForceModel.Relativity import RelativityIERS2010
from src.Auxilary.PlanetEphemerides import PlanetEphemerides
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat
import numpy as np


def demo1():
    fr = Frame(EOP(True).load())
    fr.setTime(55682.000000000000, TimeFormat.UTC_MJD)

    ep = PlanetEphemerides().configure().setPlanets()
    planets = ep.getSunPosAndVel(fr)

    re = RelativityIERS2010()
    re.setConstant().setKind().setSunState(*planets).\
        setSatState(pos=np.array([100000, 100000, 200000]) + 6378136.3e0, vel=np.array([3000, 5000, 6000]))

    print(re.getAcceleration())


if __name__ == '__main__':
    demo1()

