from src.Solver.NEQ import GravityNEQperArc
from src.Preference.EnumType import SatID


def demo():
    neq = GravityNEQperArc(0)
    neq.configure().get_neq()


if __name__ == '__main__':
    demo()

