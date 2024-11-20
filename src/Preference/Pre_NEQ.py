import json


class NEQConfig:
    def __init__(self):
        self.OrbitAConfig = self.OrbitA().__dict__
        self.OrbitBConfig = self.OrbitB().__dict__
        self.SSTConfig = self.SST().__dict__
        self.PathOfFilesConfig = self.PathOfFiles().__dict__

    class OrbitA:
        def __init__(self):
            self.IsRequired = True
            self.UseKinematicOrbit = False
            self.FixKinematicOrbit = False

    class OrbitB:
        def __init__(self):
            self.IsRequired = True
            self.UseKinematicOrbit = False
            self.FixKinematicOrbit = False

    class SST:
        def __init__(self):
            self.IsRequired = True
            self.measurement = 2
            self.CalibrateEmpiricalParameters = False

    class PathOfFiles:
        def __init__(self):
            self.NormalEqTemp = "../temp/NEQ"
            self.ResultCS = "../result/SH"


def class_to_json():
    Obj1 = NEQConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/NEQConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()

