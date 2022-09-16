from src.Preference.EnumType import SingleStepType, MultiStepType


class SingleStepConfig:
    stepsize = 5
    Npoints = 10
    isRecord = True
    SingleStepOde = SingleStepType.RKN


class MultiStepConfig:
    stepsize = 5
    Npoints = 10
    order = 8
    isRecord = True
    iterNum = 10
    rltot = 1e-13
    MultiStepOde = MultiStepType.GaussJackson


class KO:
    stepsize = 30
    Npoints = 720 * 2
    isRecord = True
    GM = 398600.4415
    rltot = 1e-9

