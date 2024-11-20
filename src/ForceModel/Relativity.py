from src.ForceModel.BaseSecondDerivative import BaseSecondDerivative
from src.Preference.EnumType import RelativityType
import numpy as np
from src.Preference.Pre_ForceModel import ForceModelConfig


class AbsRelativity(BaseSecondDerivative):

    def __init__(self):
        super(AbsRelativity, self).__init__()
        self._gamma = None
        self._beta = None
        self._J = None
        self.kind = None

    def setSatState(self, pos: np.ndarray, vel: np.ndarray):
        pass

    def setConstant(self, GM_Eatrh, c_light, GM_Sun):
        pass

    def setSunState(self, pos: np.ndarray, vel: np.ndarray):
        pass

    def getAcceleration(self):
        pass

    def getSchwarzChild(self):
        pass

    def getLenseThirring(self):
        pass

    def getDesitter(self):
        pass


class RelativityIERS2010(AbsRelativity):

    def __init__(self):
        super(RelativityIERS2010, self).__init__()
        self.__GM_Sun = None
        self.__GM_Earth = None
        self.__C_Light = None
        self.__satPos = None
        self.__satVel = None
        self.__sunState = None
        self.__FMConfig = None
        self._RelativityConfig = None

    def configure(self, FMConfig: ForceModelConfig):
        self.__FMConfig = FMConfig

        '''config relativiy'''
        self._RelativityConfig = self.__FMConfig.Relativity()
        self._RelativityConfig.__dict__.update(self.__FMConfig.RelativityConfig.copy())

        self.__GM_Earth = self._RelativityConfig.GM_Earth
        self.__GM_Sun = self._RelativityConfig.GM_Sun
        self.__C_Light = self._RelativityConfig.C_Light
        self._gamma = self._RelativityConfig.Gamma
        self._beta = self._RelativityConfig.Beta
        self._J = self._RelativityConfig.J
        kind = self._RelativityConfig.kind
        self.kind = [key for key, value in kind.items() if value]
        return self

    def setSatState(self, pos: np.ndarray, vel: np.ndarray):
        self.__satPos = pos
        self.__satVel = vel

        return self

    def setSunState(self, pos: np.ndarray, vel: np.ndarray):
        self.__sunState = [pos, vel]
        return self

    def getAcceleration(self):
        """
        calculate acceleration [m/s^2] by defining terms
        :return:
        """
        acc = np.zeros(3)
        if RelativityType.SchwarzChild.name in self.kind:
            acc += self.getSchwarzChild()

        if RelativityType.LenseThirring.name in self.kind:
            acc += self.getLenseThirring()

        if RelativityType.Desitter.name in self.kind:
            acc += self.getDesitter()

        return acc

    def getSchwarzChild(self):
        """
        calculate acceleration [m/s^2]
        :return: SchwarzChild correction to acceleration
        """
        r = self.__satPos
        v = self.__satVel
        rdotv = np.dot(r, v)

        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)

        acc = self.__GM_Earth / (self.__C_Light ** 2 * r_norm ** 3) * \
              ((2 * (self._beta + self._gamma) * self.__GM_Earth / r_norm
                - self._gamma * v_norm ** 2) * r +
               2 * (1 + self._gamma) * rdotv * v)
        return acc

    def getLenseThirring(self):
        """

        :return: Lense-Thirring correction to acceleration
        """
        r = self.__satPos
        v = self.__satVel
        r_norm = np.linalg.norm(r)

        acc = 2 * self.__GM_Earth / (self.__C_Light ** 2 * r_norm ** 3) * (
                3 / r_norm ** 2 * (np.cross(r, v)) * np.dot(r, self._J) +
                np.cross(v, self._J)
        )

        return acc

    def getDesitter(self):
        """

        :return: De sitter correction to acceleration
        """

        rs, vs = self.__sunState
        vel = self.__satVel

        term1 = self.__GM_Sun / self.__C_Light ** 2 / np.linalg.norm(rs) ** 3

        acc = 3 * np.cross(np.cross(-vs, term1 * rs), vel)

        return acc

