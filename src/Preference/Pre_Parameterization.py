import src.Preference.EnumType as EnumType


class TransitionMatrix:
    isRequired = True
    Parameter_Number = 6
    AllisGlobal = False


class Accelerometer:
    isRequired = False
    Parameter_Number = 9
    AllisGlobal = False


class StokesCoefficients:
    isRequired = False
    Parameter_Number = 8277
    AllisGlobal = True
    SortMethod = EnumType.CSseq.SortByOrder
    MaxDegree = 90
    MinDegree = 2

