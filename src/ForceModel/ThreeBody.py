from src.ForceModel.BaseSecondDerivative import BaseSecondDerivative
import numpy as np
from src.Frame.Frame import Frame
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Preference.Pre_ForceModel import ForceModelConfig


class AbsThreeBody(BaseSecondDerivative):

    def __init__(self):
        super(AbsThreeBody, self).__init__()
        self._Planets = None
        self._PlanetPos = None
        self._sat = None
        self._GM = None
        self._FMConfig = None
        self._ThreeBodyConfig = None

    def configure(self, FMConfig: ForceModelConfig):
        self._FMConfig = FMConfig
        '''config ThreeBody'''
        self._ThreeBodyConfig = self._FMConfig.ThreeBody()
        self._ThreeBodyConfig.__dict__.update(self._FMConfig.ThreeBodyConfig.copy())
        self._GM = self._ThreeBodyConfig.ThreeBody_GM
        planets = self._ThreeBodyConfig.include_planets
        self._Planets = [key for key, value in planets.items() if value]
        return self

    def setPlanetPos(self, planetPos: dict):
        """

        :param planetPos: a dictionary data set that contains key (name of the planet) and value (position of the planet),
                        unit should be given in [m], GCRS frame
        :return:
        """
        self._PlanetPos = planetPos

        assert set(planetPos.keys()) <= set(self._GM.keys())

        return self

    def setSatPos(self, pos):
        """

        :param pos: the position of satellite to be compute the force, unit [m], GCRS frame
        :return:
        """
        self._sat = pos
        return self

    def getAcceleration(self):
        return None

    def get_centralforce(self):
        """
        Acc caused by the Earth for test use
        :return:
        """
        sat = self._sat
        norm = np.linalg.norm(np.array(sat))
        acc = -self._GM['Earth'] * np.array(sat) / norm ** 3
        return acc

    def getDaDx(self):
        """
        A SYMMETRIC matrix is returned as below
                ax/dx ay/dx az/dx         ax/dx ax/dy ax/dz
                ...   ay/dy az/dy  =            ay/dy ay/dz
                ...   ...   az/dz                     az/dz
        Compute the partial derivative of acceleration with respect to the position
        :return:da_x/dx da_y/dy da_z/dz = Vxx, Vxy, Vxz , Vyx, Vyy, Vyz , Vzx, Vzy, Vzz
        """

        du2xyz_final = np.zeros((3, 3))

        PlanetPos = self._PlanetPos
        sat = self._sat

        assert PlanetPos is not None and sat is not None

        for planet in self._Planets:
            '''Relative position vector of satellite w.r.t. point mass '''
            s = np.array(PlanetPos[planet])
            d = np.array(sat) - s
            if planet == 'Earth':
                d = np.array(sat)

            norm_d = np.linalg.norm(d)
            d = d.reshape((3, 1)).astype(np.float64)
            du2xyz_final += -self._GM[planet] * (1. / norm_d ** 3 * np.identity(3) -
                                                  3. * np.matmul(d, d.T) / norm_d ** 5)

            pass

        return du2xyz_final


class ThreeBody(AbsThreeBody):

    def __init__(self):
        super(ThreeBody, self).__init__()

    def getAcceleration(self):
        """

        :return: acceleration in GCRS frame
        """

        planets = self._PlanetPos
        sat = self._sat

        assert planets is not None and sat is not None

        acc = np.zeros(3)

        for planet in self._Planets:
            '''Relative position vector of satellite w.r.t. point mass '''
            if planet == 'Earth':
                norm = np.linalg.norm(np.array(sat))
                acc += -self._GM[planet] * np.array(sat) / norm ** 3
                continue

            s = np.array(planets[planet])
            d = np.array(sat) - s
            norm_d = np.linalg.norm(d)
            norm_s = np.linalg.norm(s)
            acc += -self._GM[planet] * (d / norm_d ** 3 + s / norm_s ** 3)
            pass

        return acc


class ThreeBodyPlusJ2(AbsThreeBody):

    def __init__(self):
        super(ThreeBodyPlusJ2, self).__init__()

    def get_indirectJ2(self):
        """
        This effect is generated when considering the Earth as an ellipsoid. However, this effect is quite a minor
        quantity that can be neglected in practice.
        This could be also computed with gravity field of Earth (only C20) by entering the moon and Sun positions.
        Notice: this asks for the position of Sun and Moon in unit of [meters]
        :param rm: rotation matrix from inertial frame to earth-fixed frame, which can be acquired from 'Frame' class
        :return: acceleration due to J2 effect caused by Sun and Moon, [GCRS frame]
        """
        rm = self._time.getRotationMatrix
        acc = np.zeros(3)

        RE = 6378136.6
        J2 = -4.84166854896119e-04 * (-np.sqrt(5))  # 1.0826359e-3

        # assert 'Sun' in self._planet.keys()
        # assert 'Moon' in self._planet.keys()
        '''intersection'''
        target = {'Sun', 'Moon'} & set(self._Planets)

        for planet in target:
            '''get pos in GCRS frame'''
            pos = np.array(self._PlanetPos[planet])

            '''get pos in ITRS frame'''
            pos = Frame.PosGCRS2ITRS(pos, rm)

            '''get pos in local geographical frame, please note the sign!'''
            lon, lat, r = GeoMathKit.CalcPolarAngles(-pos)

            '''below is a spherical harmonic expansion up to degree and order 2. '''
            mu = 1.5 * J2 * self._GM[planet] * RE ** 2 / r ** 4
            term1 = np.array([-1, -1, -3])
            term2 = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

            '''get acc in ITRS'''
            acc -= mu * (5 * np.sin(lat) ** 2 + term1) * term2

            pass

        '''get acc in GCRS'''
        acc = Frame.AccITRS2GCRS(acc, rm).flatten()

        return acc

    def getAcceleration(self):
        """

        :return: acceleration in GCRS frame
        """
        planets = self._PlanetPos
        sat = self._sat

        assert planets is not None and sat is not None

        acc = np.zeros(3)

        for planet in self._Planets:
            '''Relative position vector of satellite w.r.t. point mass '''
            if planet == 'Earth':
                norm = np.linalg.norm(np.array(sat))
                acc += -self._GM[planet] * np.array(sat) / norm ** 3
                continue

            s = np.array(planets[planet])
            d = np.array(sat) - s
            norm_d = np.linalg.norm(d)
            norm_s = np.linalg.norm(s)
            acc += -self._GM[planet] * (d / norm_d ** 3 + s / norm_s ** 3)
            pass
        acc += self.get_indirectJ2()
        return acc


