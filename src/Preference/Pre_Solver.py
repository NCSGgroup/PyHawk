from src.Preference.EnumType import SSTObserve
import json


class SolverConfig:
    def __init__(self):
        self.SSTOption = SSTObserve.RangeRate.name
        self.OrbitKinFactor = 1e11
        self.InitialOrbitWeighting = "Unit"
        self.GLSResidualSpace = "Physical"
        self.OrbitFre = 8
        self.OrbitSamplingSeconds = 60.0
        self.OrbitSamplingOrigin = "GPS"
        self.CovariancePoolingMode = "Legacy"
        self.ArcWeightingConfig = {"Mode": "Off"}
        self.CovarianceGapMode = "InterpolatePSD"
        self.CovariancePositiveDefiniteMode = "DCTClip"
        self.CovarianceEpochMapping = "LegacyIndex"
        self.CovarianceMaxLagSeconds = None
        self.CovarianceZeroFrequencyPolicy = "Keep"
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

