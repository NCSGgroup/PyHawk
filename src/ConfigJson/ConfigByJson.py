import numpy as np

import src.Preference.EnumType as EnumType
import src.Preference.backups.Pre_ForceModel as Pre_ForceModel


class ConfigFMbyJson:

    def __init__(self):
        self.__config = None

    # print(EnumType.InstanceOceanTide.EOT11a)

    def configure(self, config: dict):
        self.__config = config
        for term in self.__config.keys():
            content = self.__config[term]
            if term == 'include_ForceType':
                for force in EnumType.ForceType:
                    Pre_ForceModel.include_ForceType[force] = content[force.name]
            elif term == 'include_GravHarType':
                for GravHarType in EnumType.GravHarType:
                    Pre_ForceModel.include_GravHarType[GravHarType] = content[GravHarType.name]
            elif term == 'instance_GravHar':
                for GravHar in EnumType.GravHarType:
                    Pre_ForceModel.instance_GravHar[GravHar] = content[GravHar.name]
            elif term == 'Relativity':
                Pre_ForceModel.RelativityConfig.J = np.array(content['J'])
            elif term == 'LoveNumberConfig':
                Pre_ForceModel.LoveNumberConfig.method = EnumType.LoveNumberType[content['method']]
            elif term == 'PoleTideConfig':
                Pre_ForceModel.PoleTideConfig.LoveNumberType = Pre_ForceModel.LoveNumberConfig.method
        return self


