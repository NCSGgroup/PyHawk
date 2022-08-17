import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.ForceModel.AtmosTide import AtmosTide
from src.ForceModel.NonTide import NonTide
from src.ForceModel.OceanTide import OceanTide
from src.ForceModel.PoleTide import PoleTide
from src.ForceModel.SolidTide import SolidTide
from src.Preference.EnumType import ForceType


class ConfigModel:
    def __init__(self):
        self.include_ForceType = None
        pass

    def configure(self, **kwargs):
        include_ForceType = kwargs.get('include_ForceType', Pre_ForceModel.include_ForceType)
        self.include_ForceType = [key for key, value in include_ForceType.items() if value]

        if ForceType.PoleTide in self.include_ForceType:
            PoleTide().accumulate()
        if ForceType.OceanTide in self.include_ForceType:
            PoleTide().accumulate()


