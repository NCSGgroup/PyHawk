import pathlib as pathlib
import src.Preference.EnumType as EnumType


class RangeRate:
    ParameterFitting = EnumType.SSTFitOption.biasC0C1_OneCPR
    OutlierTimes = 3
    OutlierLimit = 7e-07
    ArcLength = 1.5
    Iterations = 2


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent

    # ======================= For the Adjust===============================
    StateVectorDataTemp = root_dir.joinpath('temp', 'StateVector')
    OrbitAdjustResTemp = root_dir.joinpath('temp', 'OrbitAdjustment')
    RangeRateTemp = root_dir.joinpath('temp', 'RangeRate')
