import src.Preference.EnumType as EnumType
import src.Preference.Pre_ForceModel as Pre_ForceModel


class ConfigFMbyJson:

    def __init__(self):
        self.__config = None

    def configure(self, config: dict):
        self.__config = config
        for term in self.__config.keys():
            content = self.__config[term]
            if term == 'include_ForceType':
                for force in EnumType.ForceType:
                    Pre_ForceModel.include_ForceType[force] = content[force.name]
            elif term == 'include_GravHarType':
                for GravHar in EnumType.GravHarType:
                    Pre_ForceModel.include_GravHarType[GravHar] = content[GravHar.name]
        return self


