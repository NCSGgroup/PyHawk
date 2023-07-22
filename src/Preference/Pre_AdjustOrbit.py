import src.Preference.EnumType as EnumType
import json


class AdjustOrbitConfig:
    def __init__(self):
        self.RangeRateConfig = self.RangeRate().__dict__
        self.PathOfFilesConfig = self.PathOfFiles().__dict__

    class RangeRate:
        def __init__(self):
            self.ParameterFitting = EnumType.SSTFitOption.twice_biasC0C1_OneCPR.name
            self.OutlierTimes = 3  # how
            self.OutlierLimit = 1e-07
            self.ArcLength = 1.5
            self.Iterations = 2  # times

    class PathOfFiles:
        def __init__(self):
            # ======================= For the Adjust===============================
            self.StateVectorDataTemp = "../temp/StateVector"
            self.OrbitAdjustResTemp = "../temp/OrbitAdjustment"
            self.RangeRateTemp = "../temp/RangeRate"


def class_to_json():
    Obj1 = AdjustOrbitConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/AdjustOrbitConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()
