import pathlib as pathlib
import src.Preference.EnumType as EnumType


class RangeRate:
    ParameterFitting = EnumType.SSTFitOption.biasC0C1_OneCPR
    OutlierTimes = 3 #how
    OutlierLimit = 5e-05
    ArcLength = 1.5
    Iterations = 2 #times


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent.parent

    # ======================= For the Adjust===============================
    StateVectorDataTemp = root_dir.joinpath('temp', 'StateVector')
    OrbitAdjustResTemp = root_dir.joinpath('temp', 'OrbitAdjustment')
    RangeRateTemp = root_dir.joinpath('temp', 'RangeRate')

