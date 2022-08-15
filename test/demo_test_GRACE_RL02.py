from src.Frame.Frame import Frame, EOP
from src.Interface.L1b import GRACE_RL02
def demo1():
    fr = Frame(EOP().load())
    s = GRACE_RL02().configure().setDate()
    # print(s.getDate())
    s.read_double_sat()
    # s.getData()
    # s.clear()
    pass


if __name__ == '__main__':
    demo1()