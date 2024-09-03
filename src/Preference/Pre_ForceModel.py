# -*- coding: utf-8 -*-
# @Author  : wuyi
# @Time    : 2023/3/23 11:06
# @File    : Pre_ForceModel.py
# @Software: PyCharm
import json
import src.Preference.EnumType as EnumType
import pathlib as pathlib
import numpy as np


class ForceModelConfig:
    def __init__(self):
        self.include_ForceType = {EnumType.ForceType.NonConserveForce.name: True,
                                  EnumType.ForceType.GravHar.name: True,
                                  EnumType.ForceType.Relativity.name: True,
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

        self.NonConservativeConfig = self.NonConservative().__dict__

        self.LoveNumberConfig = self.LoveNumber().__dict__

        self.PoleTideConfig = self.PoleTide().__dict__

        self.SolidTideConfig = self.SolidTide().__dict__

        self.AtmosTideConfig = self.AtmosTide().__dict__

        self.OceanTideConfig = self.OceanTide().__dict__

        self.ThreeBodyConfig = self.ThreeBody().__dict__

        self.GravHarConfig = self.GravHar().__dict__

        self.NonTideConfig = self.NonTide().__dict__

        self.RefGravModelConfig = self.RefGravModel().__dict__

        self.RelativityConfig = self.Relativity().__dict__

        self.PathOfFilesConfig = self.PathOfFiles().__dict__

    class NonConservative:
        def __init__(self):
            self.date_span = ['2021-01-01', '2021-01-31']
            self.arcNo = 0
            self.sat = {EnumType.SatID.A.name: True,
                        EnumType.SatID.B.name: True}

    class LoveNumber:
        def __init__(self):
            self.Nmax = 180
            self.method = EnumType.LoveNumberType.Wang.name

    class SolidTide:
        def __init__(self):
            self.GM_Earth = 0.3986004415E+15
            self.GM_Sun = 1.32712442076e20
            self.GM_Moon = 0.49028010560e13
            self.Radius_Earth = 6378136.3
            self.isZeroTide = True

    class AtmosTide:
        def __init__(self):
            self.SR = 1E-3
            self.AODtide = {
                EnumType.TidesType.P1.name: True,
                EnumType.TidesType.S1.name: True,
                EnumType.TidesType.K1.name: True,
                EnumType.TidesType.N2.name: True,
                EnumType.TidesType.M2.name: True,
                EnumType.TidesType.L2.name: True,
                EnumType.TidesType.T2.name: True,
                EnumType.TidesType.S2.name: True,
                EnumType.TidesType.R2.name: True,
                EnumType.TidesType.T3.name: True,
                EnumType.TidesType.S3.name: True,
                EnumType.TidesType.R3.name: True,
            }
            self.kind = EnumType.AODtype.ATM.name
            self.Nmax = 180
            self.BB2003tide = {EnumType.TidesType.S1.name: True,
                               EnumType.TidesType.S2.name: True}

    class OceanTide:
        def __init__(self):
            self.gainFactor = 1e-3
            self.maxDegree = 120

    class ThreeBody:
        def __init__(self):
            self.ThreeBody_GM_Sun = 1.32712442076e20
            self.ThreeBody_GM = {EnumType.Planets.Earth.name: 398600.44150e9,
                                 EnumType.Planets.Sun.name: 1.32712442076e20,
                                 EnumType.Planets.Moon.name: 0.49028010560e13,
                                 EnumType.Planets.Mercury.name: 1.32712442076e20 / 6023600.0,
                                 EnumType.Planets.Venus.name: 1.32712442076e20 / 408523.71,
                                 EnumType.Planets.Mars.name: 1.32712442076e20 / 3098708.0,
                                 EnumType.Planets.Jupiter.name: 1.32712442076e20 / 1047.3486,
                                 EnumType.Planets.Saturn.name: 1.32712442076e20 / 3497.898,
                                 EnumType.Planets.Uranus.name: 5794556.465751793e9,
                                 EnumType.Planets.Neptune.name: 6836527.100580024e9,
                                 EnumType.Planets.Pluto.name: 975.5011758767654e9}
            self.include_planets = {
                EnumType.Planets.Sun.name: True,
                EnumType.Planets.Moon.name: True,
                EnumType.Planets.Mercury.name: True,
                EnumType.Planets.Venus.name: True,
                EnumType.Planets.Earth.name: True,
                EnumType.Planets.Mars.name: True,
                EnumType.Planets.Jupiter.name: True,
                EnumType.Planets.Saturn.name: True,
                EnumType.Planets.Uranus.name: False,
                EnumType.Planets.Neptune.name: False,
                EnumType.Planets.Pluto.name: False}
            self.isJ2 = True

    class GravHar:
        def __init__(self):
            self.setZero = [0, 1]
            self.degree_max = 60
            self.degree_min = 2
            self.GravHar_GM = 0.3986004415E+15
            self.GravHar_Radius = 6378136.3
            self.PosCoordinate = EnumType.Coordinate.GCRS.name
            self.ACCCoordinate = EnumType.Coordinate.GCRS.name

    class NonTide:
        def __init__(self):
            self.kind = EnumType.AODtype.GLO.name
            self.MaxDeg = 180
            self.TS = 3

    class RefGravModel:
        def __init__(self):
            self.kind = 0
            self.Nmax = 180
            self.StaticModel = EnumType.StaticGravModel.Gif48.name

    class Relativity:
        def __init__(self):
            self.GM_Sun = 1.32712442076e20
            self.GM_Earth = 3.986004415E+14
            self.C_Light = 299792458
            self.J = [0, 0, 9.8e8]
            self.Gamma = 1
            self.Beta = 1
            self.kind = {
                EnumType.RelativityType.SchwarzChild.name: True,
                EnumType.RelativityType.LenseThirring.name: True,
                EnumType.RelativityType.Desitter.name: True
            }

    class PoleTide:
        def __init__(self):
            self.Kind = {
                "Solid Earth": True,
                "Ocean": True
            }
            self.SimpleOcean = False
            self.Polar2wobble = 2
            self.MaxDegreeOfOcean = 180
            self.LoveNumberType = EnumType.LoveNumberType.Wang.name

    class PathOfFiles:
        def __init__(self):
            self.Ephemerides = "../data/ephemerides"
            self.AOD = "../data/AOD/RL06"
            self.EOT11a = "../data/eot11a"
            self.FES2014 = "../data/FES2014"
            self.Atmos = "../data/atmos"
            self.Gif48 = "../data/StaticGravityField/gif48.gfc"
            self.EIGEN6_C4 = "../data/StaticGravityField/EIGEN6-C4.gfc"
            self.GOCO02s = "../data/StaticGravityField/GOCO02s.gfc"
            self.GGM05C = "../data/StaticGravityField/GGM05C.gfc"
            self.PoleTide = "../data/poletide"
            self.LoverNumber = "../data/LoveNumber"
            self.poleTideModel = "desaiscopolecoef.txt"
            self.temp_non_conservative_data = "../temp/NonConservativeForce"


def demo1():
    Obj1 = ForceModelConfig

    dict1 = json.load(open('../../setting/ForceModelConfig.json', 'r'))
    Obj1.__dict__ = dict1

    # dict2 = json.load(open('../../setting/ForceModelConfig2.json', 'r'))
    # Obj1.__dict__ = dict2

    print(Obj1)
    print(Obj1)


def demo2():
    Obj1 = ForceModelConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/ForceModelConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    # res = ParseObjToJson(ForceModelConfig())
    # data = json.loads(res)
    # demo1()
    demo2()
