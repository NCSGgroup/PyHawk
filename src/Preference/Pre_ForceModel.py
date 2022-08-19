from src.Preference.EnumType import ForceType, GravHarType
import pathlib as pathlib

SamplePeriod = 1e-3
maxDegree = 120
GravHar_GM = 398600.44150e9
GravHar_Radius = 6378136.3e0

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


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent

    # ======================= For the ForceModel===============================
    EOT11a = root_dir.joinpath('data', 'eot11a')
    FES2014 = root_dir.joinpath('data', 'FES2014')

