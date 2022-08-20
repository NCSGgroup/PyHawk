import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.ForceModel.AtmosTide import AtmosTide
from src.ForceModel.NonTide import NonTide
from src.ForceModel.OceanTide import OceanTide
from src.ForceModel.PoleTide import PoleTide
from src.ForceModel.SolidTide import SolidTide
from src.Preference.EnumType import ForceType, GravHarType


class ConfigModel:
    def __init__(self):
        self.include_ForceType = None
        pass

    def configure(self, **kwargs):
        include_include_GravHarType = kwargs.get('include_ForceType', Pre_ForceModel.include_GravHarType)
        self.include_ForceType = [key for key, value in include_include_GravHarType.items() if value]

        if GravHarType.PoleTide in self.include_ForceType:
            PoleTide().accumulate()
        if GravHarType.OceanTide in self.include_ForceType:
            PoleTide().accumulate()


