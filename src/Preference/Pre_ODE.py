from src.Preference.EnumType import SingleStepType, MultiStepType


SingleStepOde = SingleStepType.RKN
MultiStepOde = MultiStepType.GaussJackson


stepsize = 30
Npoints = 720 * 2


class SingleStepConfig:
    isRecord = True


class MultiStepConfig:
    order = 8
    isRecord = True
    iterNum = 10
    rltot = 1e-13


class KO:
    isRecord = True
    GM = 398600.4415
    rltot = 1e-9



