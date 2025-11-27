import src.Preference.EnumType as EnumType
import json


class ParameterConfig:
    def __init__(self):
        self.TransitionMatrixConfig = self.TransitionMatrix().__dict__
        self.AccelerometerConfig = self.Accelerometer().__dict__
        self.StokesCoefficientsConfig = self.StokesCoefficients().__dict__

    class TransitionMatrix:
        def __init__(self):
            self.isRequired = True
            self.Parameter_Number = 6
            self.AllisGlobal = False

    class Accelerometer:
        def __init__(self):
            self.isRequired = True
            self.isScale = True
            self.AdjustLength = 6
            self.X_AdjustLength = 6
            self.Y_AdjustLength = 1.5
            self.Z_AdjustLength = 6
            self.Parameter_Number = 6
            self.AllisGlobal = False

    class StokesCoefficients:
        def __init__(self):
            self.isRequired = False
            self.Parameter_Number = 3717
            self.AllisGlobal = True
            self.SortMethod = EnumType.CSseq.SortByOrder.name
            self.MaxDegree = 60
            self.MinDegree = 2


def class_to_json():
    Obj1 = ParameterConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/ParameterConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()

