from src.Preference.EnumType import ForceType, GravHarType, SatID, RelativityType, LoveNumberType, AODtype, TidesType
from src.Preference.EnumType import InstanceAtmosTide, InstancePoleTide, InstanceSolidTide, InstanceOceanTide, \
    InstanceNonTide, InstanceReferenceModel
import pathlib as pathlib
import numpy as np

GravHar_GM = 398600.44150e9
GravHar_Radius = 6378136.3e0
NonConserveForce_date_span = ['2008-01-01', '2008-01-31']
arcNo = 0
sat = SatID.A

include_ForceType = {ForceType.GravHar: True,
                     ForceType.Relativity: True,
                     ForceType.NonConserveForce: False,
                     ForceType.PlanetaryPerturbation: False}

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


class OceanTideConfig:
    gainFactor = 1e-3
    maxDegree = 120


class ThreeBodyConfig:
    ThreeBody_GM_Sun = 1.32712442076e20
    ThreeBody_GM = {'Earth': 398600.44150e9,
                    'Sun': 1.32712442076e20,
                    'Moon': 0.49028010560e13,
                    'Mercury': ThreeBody_GM_Sun / 6023600.0,
                    'Venus': ThreeBody_GM_Sun / 408523.71,
                    'Mars': ThreeBody_GM_Sun / 3098708.0,
                    'Jupiter': ThreeBody_GM_Sun / 1047.3486,
                    'Saturn': ThreeBody_GM_Sun / 3497.898,
                    'Uranus': 5794556.465751793e9,
                    'Neptune': 6836527.100580024e9,
                    'Pluto': 975.5011758767654e9}


class RelativityConfig:
    GM_Sun = 1.32712442076e20
    GM_Earth = 3.986004418000000e+14
    C_Light = 299792458
    J = np.array([0, 0, 9.8e8])
    Gamma = 1
    Beta = 1
    kind = {
        RelativityType.SchwarzChild: True,
        RelativityType.LenseThirring: True,
        RelativityType.Desitter: True
    }


class EphemeridesConfig:
    isMeter = True
    include_planets = {'Sun': True,
                       'Moon': True,
                       'Mercury': True,
                       'Venus': True,
                       'Earth': False,
                       'Mars': True,
                       'Jupiter': True,
                       'Saturn': True,
                       'Uranus': True,
                       'Neptune': True,
                       'Pluto': True}


class SolidTideConfig:
    GM_Earth = 0.3986004415E+15
    GM_Sun = 1.32712442076e20
    GM_Moon = 0.49028010560e13
    Radius_Earth = 0.6378136460E+07
    isZeroTide = True


class PoleTideConfig:
    Kind = {'Solid Earth': True,
            'Ocean': True}
    SimpleOcean = False
    Polar2wobble = 2
    MaxDegreeOfOcean = 30
    LoveNumberType = LoveNumberType.Gegout97


class LoveNumberConfig:
    Nmax = 30
    method = LoveNumberType.Gegout97


class GravHarConfig:
    degree_max = 90
    degree_min = 1


class RefGravModelConfig:
    type = 0
    Nmax = 90


class NonTideConfig:
    kind = AODtype.GLO
    MaxDeg = 100
    TS = 3


class AtmosTideConfig:
    SR = 1e-3
    AODtide = {TidesType.S1: True,
               TidesType.S2: True,
               TidesType.M2: True,
               TidesType.S3: True,
               TidesType.P1: True,
               TidesType.K1: True,
               TidesType.N2: True,
               TidesType.L2: True,
               TidesType.T2: True,
               TidesType.R2: True,
               TidesType.T3: True,
               TidesType.R3: True}
    kind = AODtype.ATM
    Nmax = 120
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
    LoverNumber = root_dir.joinpath('data', 'poletide')
    poleTideModel = 'desaiscopolecoef.txt'

    Gif48 = root_dir.joinpath('data', 'StaticGravityField', 'gif48.gfc')
    AOD = root_dir.joinpath('data', 'AOD', 'RL06')
    Atmos = root_dir.joinpath('data', 'atmos')
