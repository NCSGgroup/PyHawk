

class NEQConfig:
    def __init__(self):
        self.OrbitAConfig = self.OrbitA().__dict__
        self.OrbitBConfig = self.OrbitB().__dict__
        self.SSTConfig = self.SST().__dict__
        self.PathOfFilesConfig = self.PathOfFiles().__dict__

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
        NormalEqTemp = "../temp/NEQ"

