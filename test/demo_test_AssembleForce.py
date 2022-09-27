from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import src.Preference.EnumType as EnumType
from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.Interface.ArcOperation import ArcData
import numpy as np


def demo1():

    fr = Frame(EOP().load())
    # read Arc GRACE Data
    rd = ArcData()
    res = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)

    # instance assemble model
    assem_model = Assemble2ndDerivative()
    pos = np.array([-4.151812213728250004e+06, 5.276550892813707702e+06, -1.296691103991644224e+06])
    vel = np.array([3000, 5000, 6000])
    fr.setTime(504964860, EnumType.TimeFormat.GPS_second)
    #
    assem_model = assem_model.setPosAndVel(pos, vel).setTime(fr)
    #
    # print(assem_model.getAcceleration())
    # print(assem_model.getDaDx())
    # print(assem_model.getDaDp())


if __name__ == '__main__':
    demo1()

