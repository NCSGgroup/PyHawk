import src.Preference.EnumType as EnumType


class TransitionMatrix:
    isRequired = True
    Parameter_Number = 6
    AllisGlobal = False


class Accelerometer:
    isRequired = True
    Parameter_Number = 6
    AllisGlobal = False


class StokesCoefficients:
    isRequired = True
    Parameter_Number = 8277
    AllisGlobal = True
    SortMethod = EnumType.CSseq.SortByOrder
    MaxDegree = 60
    MinDegree = 2

