import src.Preference.EnumType as EnumType


class TransitionMatrix:
    isRequired = False
    Parameter_Number = 6
    AllisGlobal = False


class Accelerometer:
    isRequired = False
    Parameter_Number = 6
    AllisGlobal = False


class StokesCoefficients:
    isRequired = False
    Parameter_Number = 3717
    AllisGlobal = True
    SortMethod = EnumType.CSseq.SortByOrder
    MaxDegree = 60
    MinDegree = 2

