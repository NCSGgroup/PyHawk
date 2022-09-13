from src.Frame.Frame import Frame, EOP
from src.Interface.L1b import GRACE_RL02, GRACE_FO_RL04
from src.Interface.GapFix import GapFix

def demo1():
    # fr = Frame(EOP().load())
    s = GRACE_RL02().configure().setDate()
    # print(s.getDate())
    s.read_double_sat()
    gf = GapFix(s).configure()
    gf.fix_all().makeReport()
    # s.getData()
    # s.clear()
    pass

if __name__ == '__main__':
    demo1()