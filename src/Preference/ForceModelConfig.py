# -*- coding: utf-8 -*-
# @Author  : wuyi
# @Time    : 2023/3/23 11:06
# @File    : ForceModelConfig.py
# @Software: PyCharm
import json

import src.Preference.EnumType as EnumType


class ForceModelConfig:
    def __init__(self):
        self._include_ForceType = {EnumType.ForceType.GravHar.name: "true",
                                   EnumType.ForceType.Relativity.name: "true",
                                   EnumType.ForceType.NonConserveForce.name: "true",
                                   EnumType.ForceType.PlanetaryPerturbation.name: "true"}

        self._instance_GravHar = {EnumType.GravHarType.NonTide.name: EnumType.InstanceNonTide.AODRL06.value,
                                  EnumType.GravHarType.PoleTide.name: EnumType.InstancePoleTide.PoleTideIERS2010.name,
                                  EnumType.GravHarType.OceanTide.name: EnumType.InstanceOceanTide.EOT11a.name,
                                  EnumType.GravHarType.AtmosphereTide.name: EnumType.InstanceAtmosTide.AtmosAOD.name,
                                  EnumType.GravHarType.SolidEarthTide.name: EnumType.InstanceSolidTide.SolidTideIERS2010.name,
                                  EnumType.GravHarType.ReferenceModel.name: EnumType.InstanceReferenceModel.RefGravModel.name}

        # self._include_GravHarType = {EnumType.GravHarType.NonTide: True,
        #                              EnumType.GravHarType.PoleTide: True,
        #                              EnumType.GravHarType.OceanTide: True,
        #                              EnumType.GravHarType.AtmosphereTide: True,
        #                              EnumType.GravHarType.SolidEarthTide: True,
        #                              EnumType.GravHarType.ReferenceModel: True}
        #



def demo1():
    Obj1 = ForceModelConfig()
    dict1 = json.load(open('ForceModelConfig.json', 'r'))
    Obj1.__dict__ = dict1

    Obj2 = ForceModelConfig()
    dict2 = json.load(open('ForceModelConfig2.json', 'r'))
    Obj2.__dict__ = dict2

    print(Obj1._include_ForceType)
    print(Obj2._include_ForceType)


def demo2():
    Obj1 = ForceModelConfig()
    tc_dict = vars(Obj1)
    with open('ForceModelConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    # res = ParseObjToJson(ForceModelConfig())
    # data = json.loads(res)
    demo1()
    # demo2()
