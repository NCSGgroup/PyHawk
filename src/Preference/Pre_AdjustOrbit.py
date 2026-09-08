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
            self.ArcOutlierEnabled = True
            self.ArcOutlierSignificance = 0.01
            self.ArcOutlierMaxFraction = 0.1
            self.ArcOutlierMinArcs = 8
            self.ArcOutlierTrimFraction = 0.01

    class RangeRate:
        def __init__(self):
            self.ParameterFitting = EnumType.SSTFitOption.biasC0C1_OneCPR.name
            self.OutlierTimes = 6  # how
            self.OutlierRMS = 7e-07
            self.OutlierObs = 7e-07
            self.ArcLength = 1.5
            self.Iterations = 3  # times
            self.ArcOutlierEnabled = True
            self.ArcOutlierSignificance = 0.05
            self.ArcOutlierMaxFraction = 0.1
            self.ArcOutlierMinArcs = 8
            self.ArcOutlierMinCPRCycles = 1.5
            self.EpochOutlierEnabled = True
            self.EpochOutlierSignificance = 0.01
            self.EpochOutlierMaxFraction = 0.02
            self.EpochOutlierMinSamples = 20

    class PathOfFiles:
        def __init__(self):
            # ======================= For the Adjust===============================
            self.StateVectorDataTemp = "../temp/StateVector"
            self.OrbitAdjustResTemp = "../temp/OrbitAdjustment"
            self.RangeRateTemp = "../temp/RangeRate"
            self.FilteredRangeRateTemp = "../temp/FilteredRangeRate"
            self.StandardKBRRTemp = "../temp/StandardKBRR"
            self.LGDTemp = "../temp/LGD"
            self.StandardLGDTemp = "../temp/StandardLGD"


def class_to_json():
    Obj1 = AdjustOrbitConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/AdjustOrbitConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()
