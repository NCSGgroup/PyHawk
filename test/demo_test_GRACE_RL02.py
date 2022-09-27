from src.Frame.Frame import Frame, EOP
from src.Interface.L1b import GRACE_OLD_RL02, GRACE_NEW_RL02
import src.Preference.EnumType as EnumType

def demo1():
    fr = Frame(EOP().load())
    s = GRACE_NEW_RL02().configure().setDate()
    # print(s.getDate())
    s.read_double_sat()
    print(s.getData(EnumType.Payload.GNV, EnumType.SatID.A))
    # s.getData()
    # s.clear()
    pass


if __name__ == '__main__':
    demo1()