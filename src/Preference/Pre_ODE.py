from src.Preference.EnumType import SingleStepType, MultiStepType


class ODEConfig:
    def __init__(self):
        self.ComplexConfig = self.Complex().__dict__
        self.SingleStepConfig = self.SingleStep().__dict__
        self.MultiStepConfig = self.MultiStep().__dict__
        self.KO = self.KO().__dict__

    class Complex:
        SingleStepType = SingleStepType.RKN.name
        MultiStepType = MultiStepType.MatrixGaussJackson.name

    class SingleStep:
        stepsize = 5
        Npoints = 2156
        isRecord = True
        SingleStepOde = SingleStepType.RKN.name

    class MultiStep:
        stepsize = 5
        Npoints = 2156
        order = 8
        isRecord = True
        iterNum = 10
        rltot = 1e-13
        MultiStepOde = MultiStepType.GaussJackson.name

    class KO:
        stepsize = 30
        Npoints = 720 * 2
        isRecord = True
        GM = 398600.4415
        rltot = 1e-9

