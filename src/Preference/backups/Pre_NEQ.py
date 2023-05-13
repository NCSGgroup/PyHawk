import pathlib as pathlib


class OrbitA:
    IsRequired = True
    UseKinematicOrbit = False
    FixKinematicOrbit = False


class OrbitB:
    IsRequired = True
    UseKinematicOrbit = False
    FixKinematicOrbit = False


class SST:
    IsRequired = True
    measurement = 2
    CalibrateEmpiricalParameters = False


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent.parent
    NormalEqTemp = root_dir.joinpath('temp', 'NEQ/')
