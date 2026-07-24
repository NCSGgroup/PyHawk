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


def class_to_json():
    config = KalmanFilterConfig()
    with open("../../setting/Calibrate/KalmanFilterConfig.json", "w") as file:
        json.dump(config.__dict__, file, indent=4)


if __name__ == "__main__":
    class_to_json()
