# -*- coding: utf-8 -*-
# @Author  : wuyi
# @Time    : 2023/3/23 11:06
# @File    : ForceModelConfig.py
# @Software: PyCharm
import json

import src.Preference.EnumType as EnumType


class ForceModelConfig:
    def __init__(self):
        self._include_ForceType = {EnumType.ForceType.GravHar: True,
                                   EnumType.ForceType.Relativity: True,
                                   EnumType.ForceType.NonConserveForce: True,
                                   EnumType.ForceType.PlanetaryPerturbation: True}

        self._include_GravHarType = {EnumType.GravHarType.NonTide: True,
                                     EnumType.GravHarType.PoleTide: True,
                                     EnumType.GravHarType.OceanTide: True,
                                     EnumType.GravHarType.AtmosphereTide: True,
                                     EnumType.GravHarType.SolidEarthTide: True,
                                     EnumType.GravHarType.ReferenceModel: True}

        self._instance_GravHar = {EnumType.GravHarType.NonTide: EnumType.InstanceNonTide.AODRL06,
                                  EnumType.GravHarType.PoleTide: EnumType.InstancePoleTide.PoleTideIERS2010,
                                  EnumType.GravHarType.OceanTide: EnumType.InstanceOceanTide.EOT11a,
                                  EnumType.GravHarType.AtmosphereTide: EnumType.InstanceAtmosTide.AtmosAOD,
                                  EnumType.GravHarType.SolidEarthTide: EnumType.InstanceSolidTide.SolidTideIERS2010,
                                  EnumType.GravHarType.ReferenceModel: EnumType.InstanceReferenceModel.RefGravModel}


def ParseObjToJson(Obj):
    return Obj.__dict__.__str__().replace("\'", "\"")


if __name__ == '__main__':
    # res = ParseObjToJson(ForceModelConfig())
    # data = json.loads(res)
    # with open('ForceModelConfig.json', 'w') as f:
    #     json.dump(data, f)

    tc_dict = vars(ForceModelConfig())

    print(tc_dict)
