# -*- coding: utf-8 -*-
# @Author  : wuyi
# @Time    : 2023/3/23 11:06
# @File    : ForceModelConfig.py
# @Software: PyCharm
import json
import src.Preference.EnumType as EnumType
import pathlib as pathlib
import numpy as np


class ForceModelConfig:
    def __init__(self):
        self.include_ForceType = {EnumType.ForceType.GravHar.name: True,
                                  EnumType.ForceType.Relativity.name: True,
                                  EnumType.ForceType.NonConserveForce.name: True,
                                  EnumType.ForceType.PlanetaryPerturbation.name: True}

        self.instance_GravHar = {EnumType.GravHarType.NonTide.name: EnumType.InstanceNonTide.AODRL06.name,
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

        self.ThreeBodyConfig = {
            "ThreeBody_GM_Sun": 1.32712442076e20,
            "ThreeBody_GM": {EnumType.Planets.Earth.name: 398600.44150e9,
                             EnumType.Planets.Sun.name: 1.32712442076e20,
                             EnumType.Planets.Moon.name: 0.49028010560e13,
                             EnumType.Planets.Mercury.name: 1.32712442076e20 / 6023600.0,
                             EnumType.Planets.Venus.name: 1.32712442076e20 / 408523.71,
                             EnumType.Planets.Mars.name: 1.32712442076e20 / 3098708.0,
                             EnumType.Planets.Jupiter.name: 1.32712442076e20 / 1047.3486,
                             EnumType.Planets.Saturn.name: 1.32712442076e20 / 3497.898,
                             EnumType.Planets.Uranus.name: 5794556.465751793e9,
                             EnumType.Planets.Neptune.name: 6836527.100580024e9,
                             EnumType.Planets.Pluto.name: 975.5011758767654e9},
            "include_planets": {
                EnumType.Planets.Sun.name: True,
                EnumType.Planets.Moon.name: True,
                EnumType.Planets.Mercury.name: True,
                EnumType.Planets.Venus.name: True,
                EnumType.Planets.Earth.name: True,
                EnumType.Planets.Mars.name: True,
                EnumType.Planets.Jupiter.name: True,
                EnumType.Planets.Saturn.name: True,
                EnumType.Planets.Uranus.name: True,
                EnumType.Planets.Neptune.name: True,
                EnumType.Planets.Pluto.name: True},
            "isJ2": False
        }

        self.GravHarConfig = {
            "setZero": [0, 1],
            "degree_max": 90,
            "degree_min": 2,
            "GravHar_GM": 0.3986004415E+15,
            "GravHar_Radius": 0.6378136460E+07,
            "PosCoordinate": EnumType.Coordinate.GCRS.name,
            "ACCCoordinate": EnumType.Coordinate.GCRS.name,
        }

        self.NonTideConfig = {
            "kind": EnumType.AODtype.GLO.name,
            "MaxDeg": 180,
            "TS": 3,
        }

        self.RelativityConfig = {
            "GM_Sun": 1.32712442076e20,
            "GM_Earth": 0.3986004415E+15,
            "C_Light": 299792458,
            "J": [0, 0, 9.8e8],
            "Gamma": 1,
            "Beta": 1,
            "kind": {
                EnumType.RelativityType.SchwarzChild.name: True,
                EnumType.RelativityType.LenseThirring.name: True,
                EnumType.RelativityType.Desitter.name: True
            }
        }

        self.PathOfFilesConfig = {
            "Ephemerides": "../data/ephemerides",
            "AOD": "../data/AOD/RL06"
        }


def demo1():
    Obj1 = ForceModelConfig()

    dict1 = json.load(open('../../setting/ForceModelConfig.json', 'r'))
    Obj1.__dict__ = dict1

    dict2 = json.load(open('../../setting/ForceModelConfig2.json', 'r'))
    Obj1.__dict__ = dict2

    print(Obj1.NonConservativeConfig)
    print(Obj1.NonConservativeConfig)


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
