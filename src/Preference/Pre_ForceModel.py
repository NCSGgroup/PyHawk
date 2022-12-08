from src.Preference.EnumType import ForceType, GravHarType, SatID, \
    RelativityType, LoveNumberType, AODtype, TidesType, StaticGravModel
from src.Preference.EnumType import InstanceAtmosTide, InstancePoleTide, InstanceSolidTide, InstanceOceanTide, \
    InstanceNonTide, InstanceReferenceModel
import pathlib as pathlib
import numpy as np
import src.Preference.EnumType as EnumType


include_ForceType = {ForceType.GravHar: True,
                     ForceType.Relativity: True,
                     ForceType.NonConserveForce: True,
                     ForceType.PlanetaryPerturbation: True}

include_GravHarType = {GravHarType.NonTide: True,
                       GravHarType.PoleTide: True,
                       GravHarType.OceanTide: True,
                       GravHarType.AtmosphereTide: True,
                       GravHarType.SolidEarthTide: True,
                       GravHarType.ReferenceModel: True}

instance_GravHar = {GravHarType.NonTide: InstanceNonTide.AODRL06,
                    GravHarType.PoleTide: InstancePoleTide.PoleTideIERS2010,
                    GravHarType.OceanTide: InstanceOceanTide.EOT11a,
                    GravHarType.AtmosphereTide: InstanceAtmosTide.AtmosAOD,
                    GravHarType.SolidEarthTide: InstanceSolidTide.SolidTideIERS2010,
                    GravHarType.ReferenceModel: InstanceReferenceModel.RefGravModel}


class NonConservative:
    date_span = ['2016-01-01', '2016-01-31']
    arcNo = 0
    sat = [SatID.A]


class OceanTideConfig:
    gainFactor = 1e-3
    maxDegree = 120


class ThreeBodyConfig:
    ThreeBody_GM_Sun = 1.32712442076e20
    ThreeBody_GM = {EnumType.Planets.Earth.name: 398600.44150e9,
                    EnumType.Planets.Sun.name: 1.32712442076e20,
                    EnumType.Planets.Moon.name: 0.49028010560e13,
                    EnumType.Planets.Mercury.name: ThreeBody_GM_Sun / 6023600.0,
                    EnumType.Planets.Venus.name: ThreeBody_GM_Sun / 408523.71,
                    EnumType.Planets.Mars.name: ThreeBody_GM_Sun / 3098708.0,
                    EnumType.Planets.Jupiter.name: ThreeBody_GM_Sun / 1047.3486,
                    EnumType.Planets.Saturn.name: ThreeBody_GM_Sun / 3497.898,
                    EnumType.Planets.Uranus.name: 5794556.465751793e9,
                    EnumType.Planets.Neptune.name: 6836527.100580024e9,
                    EnumType.Planets.Pluto.name: 975.5011758767654e9}
    include_planets = {
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
        EnumType.Planets.Pluto.name: True}
    isJ2 = False


class RelativityConfig:
    GM_Sun = 1.32712442076e20
    GM_Earth = 0.3986004415E+15
    C_Light = 299792458
    J = np.array([0, 0, 9.8e8])
    Gamma = 1
    Beta = 1
    kind = {
        RelativityType.SchwarzChild: True,
        RelativityType.LenseThirring: True,
        RelativityType.Desitter: True
    }


class SolidTideConfig:
    GM_Earth = 0.3986004415E+15
    GM_Sun = 1.32712442076e20
    GM_Moon = 0.49028010560e13
    Radius_Earth = 0.6378136460E+07
    isZeroTide = False


class LoveNumberConfig:
    Nmax = 180
    method = LoveNumberType.Gegout97


class PoleTideConfig:
    Kind = {'Solid Earth': True,
            'Ocean': True}
    SimpleOcean = False
    Polar2wobble = 2
    MaxDegreeOfOcean = LoveNumberConfig.Nmax
    LoveNumberType = LoveNumberConfig.method


class GravHarConfig:
    setZero = [0, 1]
    degree_max = 90
    degree_min = 2
    GravHar_GM = 0.3986004415E+15
    GravHar_Radius = 0.6378136460E+07
    PosCoordinate = EnumType.Coordinate.GCRS
    ACCCoordinate = EnumType.Coordinate.GCRS


class SortCSConfig:
    degree_max = 90
    degree_min = 1


class RefGravModelConfig:
    kind = 0
    Nmax = 180
    StaticModel = StaticGravModel.EIGEN6_C4


class NonTideConfig:
    kind = AODtype.GLO
    MaxDeg = 180
    TS = 3


class AtmosTideConfig:
    SR = 1e-3
    AODtide = {
        TidesType.P1: True,
        TidesType.S1: True,
        TidesType.K1: True,
        TidesType.N2: True,
        TidesType.M2: True,
        TidesType.L2: True,
        TidesType.T2: True,
        TidesType.S2: True,
        TidesType.R2: True,
        TidesType.T3: True,
        TidesType.S3: True,
        TidesType.R3: True,
    }
    kind = AODtype.ATM
    Nmax = 180
    BB2003tide = {TidesType.S1: True,
                  TidesType.S2: True}


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent

    # ======================= For the ForceModel===============================
    EOT11a = root_dir.joinpath('data', 'eot11a')
    FES2014 = root_dir.joinpath('data', 'FES2014')
    temp_non_conservative_data = root_dir.joinpath('temp', 'NonConservativeForce')
    Ephemerides = root_dir.joinpath('data', 'ephemerides')
    PoleTide = root_dir.joinpath('data', 'poletide')
    LoverNumber = root_dir.joinpath('data', 'LoveNumber')
    poleTideModel = 'desaiscopolecoef.txt'

    Gif48 = root_dir.joinpath('data', 'StaticGravityField', 'gif48.gfc')
    EIGEN6_C4 = root_dir.joinpath('data', 'StaticGravityField', 'EIGEN6-C4.gfc')
    GOCO02s = root_dir.joinpath('data', 'StaticGravityField', 'GOCO02s.gfc')
    AOD = root_dir.joinpath('data', 'AOD', 'RL06')
    Atmos = root_dir.joinpath('data', 'atmos')

