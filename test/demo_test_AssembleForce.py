from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
import src.Preference.EnumType as EnumType
from src.SecDerivative.Common.Assemble2ndDerivative import Assemble2ndDerivative
from src.Interface.ArcOperation import ArcData


def demo1():

    fr = Frame(EOP().load())

    # read Arc GRACE Data
    rd = ArcData()
    res = rd.getData(arc=0, kind=EnumType.Payload.GNV, sat=EnumType.SatID.A)
    GPS_Time = res[4, 0]
    print(res[:, 1:4])
    startPos = res[4, 1:4]
    startVel = res[4, 4:7]

    # instance assemble model
    assem_model = Assemble2ndDerivative()
    assem_model = assem_model.setPosAndVel(startPos, startVel).setTime(fr.setTime(GPS_Time))
    assem_model.getDaDx()
    assem_model.getDaDp()


if __name__ == '__main__':
    demo1()

