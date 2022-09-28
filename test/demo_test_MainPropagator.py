from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import src.Preference.EnumType as EnumType
from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.ConfigJson.ConfigByJson import ConfigFMbyJson
from src.ODE.GravimetryODE import GravimetryODE
from src.Interface.ArcOperation import ArcData
import json
import numpy as np


def demo1():
    fr = Frame(EOP().load())

    # read Arc GRACE Data
    rd = ArcData()
    res = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    GPS_Time = res[4, 0]
    startPos = res[4, 1:4]
    startVel = res[4, 4:7]

    # instance assemble model
    assem_model = Assemble2ndDerivative()

    # ODE
    propagator = GravimetryODE(fr).setInitial(GPS_Time, startPos, startVel).setForceModel(assem_model)
    X_GJ, R_GJ, v = propagator.propagate()
    print(X_GJ)
    print(R_GJ)
    print(v)


if __name__ == '__main__':
    demo1()