from src.Preference.EnumType import SSTObserve
import pathlib as pathlib


class GravityInversion:
    SSTOption = SSTObserve.RangeRate


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent.parent
    DesignMatrixTemp = root_dir.joinpath('temp', 'DesignMatrixSH/')

