from src.Solver.DesignMatrix import GravityDesignMat


def demo1():
    dm = GravityDesignMat(0).configure()
    # dm.orbit_integration()
    # dm.get_orbit_design_matrix()
    dm.get_sst_design_matrix()


if __name__ == '__main__':
    demo1()

