from src.Preference.EnumType import SingleStepType, MultiStepType
import json


class ODEConfig:
    def __init__(self):
        self.ComplexConfig = self.Complex().__dict__
        self.SingleStepConfig = self.SingleStep().__dict__
        self.MultiStepConfig = self.MultiStep().__dict__
        self.KOConfig = self.KO().__dict__

    class Complex:
        def __init__(self):
            self.SingleStepType = SingleStepType.RKN.name
            self.MultiStepType = MultiStepType.MatrixGaussJackson.name

    class SingleStep:
        def __init__(self):
            self.stepsize = 5
            self.Npoints = 2156
            self.isRecord = True
            self.SingleStepOde = SingleStepType.RKN.name

    class MultiStep:
        def __init__(self):
            self.stepsize = 5
            self.Npoints = 2156
            self.order = 8
            self.isRecord = True
            self.iterNum = 10
            self.rltot = 1e-13
            self.MultiStepOde = MultiStepType.GaussJackson.name

    class KO:
        def __init__(self):
            self.stepsize = 30
            self.Npoints = 720 * 2
            self.isRecord = True
            self.GM = 398600.4415
            self.rltot = 1e-9


def class_to_json():
    Obj1 = ODEConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/ODEConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()

