from src.Solver.DesignMatrix import GravityDesignMat


def demo1():
    dm = GravityDesignMat(0)
    dm.orbit_integration()


if __name__ == '__main__':
    demo1()

