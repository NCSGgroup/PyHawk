import src.Preference.EnumType as EnumType
import src.Preference.Pre_ForceModel as Pre_ForceModel
from src.ForceModel.Assemble2ndDerivative import Assemble2ndDerivative
from src.ODE.ComplexODE import ComplexODE, ConfigComplexODE
from src.Preference.EnumType import SingleStepType, MultiStepType
import src.Preference.Pre_ODE as Pre_ODE
from src.Preference.EnumType import TimeFormat, GravHarType
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP


class ConfigFMbyJson:

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

    def __init__(self, fr: Frame):
        self.__fr = fr
        self.__t0 = None
        self.__r0 = None
        self.__v0 = None
        self.__force = None
        pass

    def setInitial(self, t0=None, r0=None, v0=None):
        self.__t0 = t0
        self.__r0 = r0
        self.__v0 = v0
        return self

    def setForceModel(self, force: Assemble2ndDerivative):
        self.__force = force
        return self

    def forceModel(self, t, r, v):
        self.__force = self.__force.setPosAndVel(r, v).setTime(self.__fr.setTime(t, TimeFormat.GPS_second))
        return self.__force.getAcceleration()

    def propagate(self):
        config = ConfigComplexODE(SingleStepType.RKN, MultiStepType.GaussJackson).setOdePar()
        propagator = ComplexODE(config).setSecondDerivative(self.forceModel).setInitial(self.__t0, self.__r0, self.__v0)
        return propagator.propagate()


