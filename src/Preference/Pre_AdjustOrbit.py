import pathlib as pathlib


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent

    # ======================= For the Adjust===============================
    StateVectorDataTemp = root_dir.joinpath('temp', 'StateVector')
    OrbitAdjustResTemp = root_dir.joinpath('temp', 'OrbitAdjustment')

