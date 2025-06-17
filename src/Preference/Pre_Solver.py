from src.Preference.EnumType import SSTObserve
import json


class SolverConfig:
    def __init__(self):
        self.SSTOption = SSTObserve.RangeRate.name
        self.OrbitKinFactor = 1e11
        self.DesignMatrixTemp = "../temp/DesignMatrixSH"
        self.CovarianceMatrix = "../temp/CovarianceMatrix"
        self.PostParameter = "../temp/PostParameter"
        self.PostKBRR = "../temp/PostKBRR"
        self.PostOrbit = "../temp/PostOrbit"


def class_to_json():
    Obj1 = SolverConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/SolverConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()

