# -*- coding: utf-8 -*-
# @Author  : wuyi
# @Time    : 2023/3/23 11:06
# @File    : ForceModelConfig.py
# @Software: PyCharm
import json
import src.Preference.EnumType as EnumType
import pathlib as pathlib


class ForceModelConfig:
    def __init__(self):
        root_dir = pathlib.Path(__file__).parent.parent.parent,
        self.include_ForceType = {EnumType.ForceType.GravHar.name: True,
                                  EnumType.ForceType.Relativity.name: True,
                                  EnumType.ForceType.NonConserveForce.name: True,
                                  EnumType.ForceType.PlanetaryPerturbation.name: True}

        self.instance_GravHar = {EnumType.GravHarType.NonTide.name: EnumType.InstanceNonTide.AODRL06.value,
                                 EnumType.GravHarType.PoleTide.name: EnumType.InstancePoleTide.PoleTideIERS2010.name,
                                 EnumType.GravHarType.OceanTide.name: EnumType.InstanceOceanTide.EOT11a.name,
                                 EnumType.GravHarType.AtmosphereTide.name: EnumType.InstanceAtmosTide.AtmosAOD.name,
                                 EnumType.GravHarType.SolidEarthTide.name: EnumType.InstanceSolidTide.SolidTideIERS2010.name,
                                 EnumType.GravHarType.ReferenceModel.name: EnumType.InstanceReferenceModel.RefGravModel.name}

        self.include_GravHarType = {EnumType.GravHarType.NonTide.name: True,
                                    EnumType.GravHarType.PoleTide.name: True,
                                    EnumType.GravHarType.OceanTide.name: True,
                                    EnumType.GravHarType.AtmosphereTide.name: True,
                                    EnumType.GravHarType.SolidEarthTide.name: True,
                                    EnumType.GravHarType.ReferenceModel.name: True}

        self.NonConservativeConfig = {"date_span": ['2008-01-01', '2008-01-31'],
                                      "arcNo": 0,
                                      "sat": {EnumType.SatID.A.name: True,
                                              EnumType.SatID.B.name: True}}

        self.OceanTideConfig = {"gainFactor": 1e-3,
                                "maxDegree": 120}

        self.PathOfFilesConfig = {
            "Ephemerides": root_dir.joinpath('data', 'ephemerides')
        }


def demo1():
    Obj1 = ForceModelConfig()
    dict1 = json.load(open('../../setting/ForceModelConfig.json', 'r'))
    Obj1.__dict__ = dict1

    Obj2 = ForceModelConfig()
    dict2 = json.load(open('../../setting/ForceModelConfig2.json', 'r'))
    Obj2.__dict__ = dict2

    print(Obj1.NonConservativeConfig)
    print(Obj2.NonConservativeConfig)


def demo2():
    Obj1 = ForceModelConfig()
    tc_dict = vars(Obj1)
    with open('../../setting/ForceModelConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    # res = ParseObjToJson(ForceModelConfig())
    # data = json.loads(res)
    # demo1()
    demo2()
