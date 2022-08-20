from src.Preference.EnumType import ForceType, GravHarType, SatID, RelativityType
import pathlib as pathlib

SamplePeriod = 1e-3
maxDegree = 120
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
                     GravHarType.ReferenceModel: True,}

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
ThreeBody_J2 = False


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent

    # ======================= For the ForceModel===============================
    EOT11a = root_dir.joinpath('data', 'eot11a')
    FES2014 = root_dir.joinpath('data', 'FES2014')
    temp_non_conservative_data = root_dir.joinpath('temp', 'NonConservativeForce')

