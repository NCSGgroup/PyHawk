from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat, GravHarType
from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
import numpy as np


def demo1():
    assem_model = Assemble2ndDerivative()
    fr = Frame(EOP(True).load())
    fr.setTime(504964860, TimeFormat.GPS_second)

    pos = np.array([-4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06])
    vel = np.array([3000, 5000, 6000])

    assem_model = assem_model.setPosAndVel(pos, vel).setTime(fr)

    # print(assem_model.getAcceleration())
    # print(assem_model.getDaDx())
    print(assem_model.getDaDp())


if __name__ == '__main__':
    demo1()

