import src.Preference.EnumType as EnumType
import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.ForceModel.Assemble2ndDerivative import AssembleForce
import src.Preference.Pre_ODE as Pre_ODE
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP


class PropagatorConfig:

    def __init__(self):
        self.__config = None

    def configure(self, config: dict):
        self.__config = config
        for term in self.__config.keys():
            content = self.__config[term]
            if term == 'include_ForceType':
                Pre_ForceModel.include_ForceType[EnumType.ForceType.GravHar] = content['GravHar']
                Pre_ForceModel.include_ForceType[EnumType.ForceType.NonConserveForce] = content['NonConserveForce']
                Pre_ForceModel.include_ForceType[EnumType.ForceType.Relativity] = content['Relativity']
                Pre_ForceModel.include_ForceType[EnumType.ForceType.PlanetaryPerturbation] = content['PlanetaryPerturbation']
            elif term == 'include_GravHarType':
                Pre_ForceModel.include_GravHarType[EnumType.GravHarType.NonTide] = content['NonTide']
                Pre_ForceModel.include_GravHarType[EnumType.GravHarType.PoleTide] = content['PoleTide']
                Pre_ForceModel.include_GravHarType[EnumType.GravHarType.OceanTide] = content['OceanTide']
                Pre_ForceModel.include_GravHarType[EnumType.GravHarType.AtmosphereTide] = content['AtmosphereTide']
                Pre_ForceModel.include_GravHarType[EnumType.GravHarType.SolidEarthTide] = content['SolidEarthTide']
                Pre_ForceModel.include_GravHarType[EnumType.GravHarType.ReferenceModel] = content['ReferenceModel']
        return self


class Propagator:

    def __init__(self):
        self.__fr = Frame(EOP(True).load())
        pass

    def setInitial(self, t0=None, r0=None, v0=None):

        return self

    def setForceModel(self, force: AssembleForce):
        return self

    def propagate(self, stepsize=Pre_ODE.SingleStepConfig.stepsize, step=Pre_ODE.SingleStepConfig.Npoints):
        pass


