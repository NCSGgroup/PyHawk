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


class RKOrder(Enum):
    rk4 = 1
    rk8 = 2
    rkn = 3


