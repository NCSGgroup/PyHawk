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


class Coordinate(Enum):
    GCRS = 0
    ITRS = 1


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
    MatrixRKN = 4


class MultiStepType(Enum):
    GaussJackson = 1
    MatrixGaussJackson = 2
    ABM8 = 3


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


class InstanceOceanTide(Enum):
    EOT11a = 0
    FES2014 = 1


class InstancePoleTide(Enum):
    PoleTideIERS2010 = 0


class InstanceAtmosTide(Enum):
    AtmosBB2003 = 0
    AtmosAOD = 1


class InstanceSolidTide(Enum):
    SolidTideIERS2010 = 0


class InstanceNonTide(Enum):
    AODRL06 = 0


class InstanceReferenceModel(Enum):
    RefGravModel = 0


class RelativityType(Enum):
    SchwarzChild = 0
    LenseThirring = 1
    Desitter = 2


class JPLempheric(Enum):
    DE405 = 1
    DE421 = 2
    DE430 = 3
    DE440 = 4


class LoveNumberType(Enum):
    PREM = 1
    AOD04 = 2
    Wang = 3
    IERS = 4
    Gegout97 = 5


class AODtype(Enum):
    ATM = 0
    OCN = 1
    GLO = 2
    OBA = 3


class TidesType(Enum):
    S1 = 0
    S2 = 1
    S3 = 2
    M2 = 3
    P1 = 4
    K1 = 5
    N2 = 6
    L2 = 7
    T2 = 8
    R2 = 9
    T3 = 10
    R3 = 11


class StaticGravModel(Enum):
    Gif48 = 0
    EIGEN6_C4 = 1
    GOCO02s = 2
    GGM05C = 3


class CSseq(Enum):
    Normal = 1
    SortByOrder = 2
    SortByDegree = 3
    kind4 = 4


class ParaType(Enum):
    """
    ALL the parameters are sorted with this order
    """
    TransitionMatrix = 1
    Accelerometer = 2
    StokesCoefficients = 3


class ParaField(Enum):
    Global = 1
    Local = 2


class Planets(Enum):
    Sun = 10
    Moon = 11
    Mercury = 1
    Venus = 2
    Earth = 3
    Mars = 4
    Jupiter = 5
    Saturn = 6
    Uranus = 7
    Neptune = 8
    Pluto = 9


class SSTFitOption(Enum):
    biasC0C1 = 1
    biasC0C1C2 = 2
    biasC0C1C2C3 = 3
    biasC0_OneCPR = 4
    biasC0C1_OneCPR = 5
    biasC0C1C2_OneCPR = 6
    twice_biasC0C1_OneCPR = 7
    twice_biasC0C1C2_OneCPR = 8
    test = 9
    biasC0C1C2C3_OneCPR = 10


class SSTObserve(Enum):
    Range = 1
    RangeRate = 2
    RangeAcc = 3


class FieldType(Enum):
    pressure = 0
    EWH = 1
    geoid = 2
    density = 3


class Level(Enum):
    L1A = 0
    L1B = 1