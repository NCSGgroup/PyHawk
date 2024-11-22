import src.Preference.EnumType as EnumType
import json


class AdjustOrbitConfig:
    def __init__(self):
        self.OrbitConfig = self.Orbit().__dict__
        self.RangeRateConfig = self.RangeRate().__dict__
        self.PathOfFilesConfig = self.PathOfFiles().__dict__

    class Orbit:
        def __init__(self):
            self.OutlierTimes = 3
            self.OutlierLimit = 1
            self.iteration = 1

    class RangeRate:
        def __init__(self):
            self.ParameterFitting = EnumType.SSTFitOption.biasC0C1_OneCPR.name
            self.OutlierTimes = 6  # how
            self.OutlierLimit = 7e-07
            self.ArcLength = 1.5
            self.Iterations = 3  # times

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
