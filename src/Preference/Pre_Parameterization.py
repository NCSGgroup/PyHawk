import src.Preference.EnumType as EnumType


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

