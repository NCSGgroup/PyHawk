import pathlib as pathlib


class RangeRate:
    ParameterFitting = 8
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
