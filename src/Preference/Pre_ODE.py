from src.Preference.EnumType import RKOrder


class RK:
    RKOrder = RKOrder.rkn
    isRecord = True
    stepsize = 0.01
    Npoints = int(3600 / stepsize)


class GJ:
    order = 8
    isRecord = True
    iterNum = 10
    rltot = 1e-13
    stepsize = 0.01
    Npoints = int(3600 / stepsize)


class KO:
    isRecord = True
    GM = 398600.4415
    rltot = 1e-9
    stepsize = 0.01
    Npoints = int(3600 / stepsize)



