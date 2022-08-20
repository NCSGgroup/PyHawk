"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2022/8/8
@Description:
"""
from enum import Enum
# ======================= For the time frame===============================


class TimeFormat(Enum):
    GPS_second = 1
    UTC_MJD = 2
    TAI_MJD = 3


class CoordinateTransform(Enum):
    CIOxy = 1
    CIOangle = 2


# ======================= For the data interface===============================
class Mission(Enum):
    GRACE_RL02 = 1
    GRACE_RL03 = 2
    GFO_RL02 = 4
    XX4_RL01 = 5
    GRACE_FO_RL04 = 6


class SatID(Enum):
    A = 1
    B = 2


class Payload(Enum):
    ACC = 1
    SCA = 2
    KBR = 3
    GNV = 4
    KinematicOrbit = 5
    LRI = 6


class SingleStepType(Enum):
    RK4 = 1
    RK8 = 2
    RKN = 3


class MultiStepType(Enum):
    GaussJackson = 1
    ABM8 = 2


class OdeVariable(Enum):
    r = 1
    drdp = 2
    drdx = 3


class ForceType(Enum):
    PlanetaryPerturbation = 0
    GravHar = 1
    NonConserveForce = 2
    Relativity = 3


class GravHarType(Enum):
    OceanTide = 0
    ReferenceModel = 1
    AtmosphereTide = 2
    SolidEarthTide = 3
    PoleTide = 4
    NonTide = 5


class RelativityType(Enum):
    SchwarzChild = 0
    LenseThirring = 1
    Desitter = 2

