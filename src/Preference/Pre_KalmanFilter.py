import json


class KalmanFilterConfig:
    """Configuration template for the standalone PyHawk Kalman adapter."""

    def __init__(self):
        self.Enabled = False
        self.AutoregressiveModelFile = ""
        self.AutoregressiveModelDataset = "model"
        self.InitialStateFile = ""
        self.InitialStateDataset = "state"
        self.InitialCovarianceFile = ""
        self.InitialCovarianceDataset = "covariance"
        self.NormalEquationFiles = []
        self.NormalDataset = "NMatrix"
        self.RightHandSideDataset = "LMatrix"
        self.LocalParameterCount = 0
        self.OutputFile = "../result/Kalman/filtered.hdf5"
        self.SaveCovariance = False
        self.SkipMissingNormalEquations = True
        self.AllowPseudoInverse = False
        self.TrainAOHIS = False
        self.AOHISInputFile = ""
        self.AOHISAOErrorDirectory = ""
        self.AOHISHydrologyDirectory = ""
        self.AOHISCryosphereDirectory = ""
        self.AOHISEpochDataset = "epochs"
        self.AOHISAOErrorDataset = "aoErr"
        self.AOHISHydrologyDataset = "H"
        self.AOHISCryosphereDataset = "I"
        self.AOHISModelOutputFile = "../result/Kalman/aohis_ar.hdf5"
        self.AOHISDegreeMax = 40
        self.AOHISDegreeMin = 2
        self.AOHISOrder = 3
        self.AOHISStateOrder = "SortByOrder"
        self.AOHISMonthlyMeanMode = "block"
        self.AOHISRemoveTrend = True
        self.AOHISRemoveAnnual = True
        self.AOHISRidge = 0.0
        self.AOHISWarmupModels = True
        self.AOHISUseWarmupModels = True
        self.AOHISWarmupDatasetPrefix = "model_order_"
        self.AOHISRequireWarmupModels = False
        self.AOHISIncludeCryosphere = True
        self.AOHISCovarianceDomain = "ewh"
        self.AOHISEWHGridFile = ""
        self.AOHISEWHLatitudeDataset = "latitude"
        self.AOHISEWHLongitudeDataset = "longitude"
        self.AOHISLandMaskDataset = "land_mask"
        self.AOHISAreaWeightsDataset = "area_weights"
        self.AOHISEWHRadius = 6378136.3
        self.AOHISEarthDensity = 5517.0
        self.AOHISWaterDensity = 1000.0
        self.AOHISLoadLoveNumbersFile = ""
        self.AOHISLoadLoveNumbersDataset = ""
        self.AOHISDistanceScaleKm = 1100.0
        self.AOHISDistanceCutoffKm = 3300.0
        self.AOHISEWHBlockSize = 256
        self.AOHISSaveResiduals = False
        self.Smooth = False
        self.Epochs = []
        self.GFCOutputDirectory = ""
        self.GFCReferenceField = ""
        self.GFCStateDataset = ""
        self.GFCDegreeMax = 40
        self.GFCDegreeMin = 2
        self.GFCStateOrder = "SortByOrder"
        self.GFCModelName = "PyHawk_Kalman"
        self.GFCRestoreReference = True
        self.GFCCovarianceDataset = ""
        self.GFCAOD1BReference = ""
        self.GFCGIAReference = ""
        self.GFCDegree1Reference = ""
        self.GFCRestoreAOD1B = True
        self.GFCRemoveGIA = True
        self.GFCRestoreDegree1 = True


def class_to_json():
    config = KalmanFilterConfig()
    with open("../../setting/Calibrate/KalmanFilterConfig.json", "w") as file:
        json.dump(config.__dict__, file, indent=4)


if __name__ == "__main__":
    class_to_json()
