import numpy as np
import src.Preference.EnumType as EnumType
from jplephem.spk import SPK
from src.Frame.Frame import Frame
from src.Preference.Pre_ForceModel import ForceModelConfig
from src.Preference.Pre_Frame import FrameConfig


class PlanetEphemerides:
    """
    This class provides the three-dimension position of all planets at solar system with DE405 or DE421.
    1.refer to https://pypi.org/project/jplephem/
    2.Validation against the fortran code has been done.
    3.The unit is kilometers
    4. All the positions are given at GCRS frame
    5. The time frame is TDB system in JD convention
    """

    def __init__(self):
        self.__meter = 1.  # In general, the unit is [km]
        self.__reset()
        self.__emphericPath = None
        self.__kernel = None
        self.__time = None
        self.__pos_planets = {}
        self.__include_plantes = {}
        self.__FMConfig = None
        self._PathConfig = None
        self._ThreeBodyConfig = None
        self.__FrameConfig = None
        self.__Ephemerides = None
        pass

    def configure(self, FMConfig: ForceModelConfig, FrameConfig: FrameConfig, emphericType=EnumType.JPLempheric.DE421):
        self.__FMConfig = FMConfig
        self.__FrameConfig = FrameConfig

        self.__Ephemerides = self.__FrameConfig.Ephemerides()
        self.__Ephemerides.__dict__.update(self.__FrameConfig.EphemeridesConfig.copy())

        self._PathConfig = self.__FMConfig.PathOfFiles()
        self._PathConfig.__dict__.update(self.__FMConfig.PathOfFilesConfig.copy())

        isMeter = self.__Ephemerides.isMeter
        if emphericType == EnumType.JPLempheric.DE405:
            filename = 'de405.bsp'
        elif emphericType == EnumType.JPLempheric.DE421:
            filename = 'de421.bsp'
        elif emphericType == EnumType.JPLempheric.DE430:
            filename = 'de430.bsp'
        elif emphericType == EnumType.JPLempheric.DE440:
            filename = 'de440.bsp'
        else:
            raise Exception

        if isMeter:
            self.__meter = 1000
        '''load plants ephemerides'''
        if self.__FMConfig.include_ForceType[EnumType.ForceType.PlanetaryPerturbation.name]:
            self._ThreeBodyConfig = self.__FMConfig.ThreeBody()
            self._ThreeBodyConfig.__dict__.update(self.__FMConfig.ThreeBodyConfig.copy())
            self.__include_plantes = self._ThreeBodyConfig.include_planets.copy()
        else:
            self.__include_plantes = self.__Ephemerides.include_planets.copy()
        # self.__include_plantes['Sun'] = True
        # self.__include_plantes['Moon'] = True
        '''set empheric path'''
        self.__emphericPath = self._PathConfig.Ephemerides + '/' + str(filename)
        self.__kernel = SPK.open(self.__emphericPath)
        return self

    def setTime(self, time: Frame):
        self.__time = time
        return self

    def __reset(self):
        self.__planets = {'Sun': 10,
                          'Moon': 11,
                          'Mercury': 1,
                          'Venus': 2,
                          'Earth': 3,
                          'Mars': 4,
                          'Jupiter': 5,
                          'Saturn': 6,
                          'Uranus': 7,
                          'Neptune': 8,
                          'Pluto': 9, }
        pass

    def setPlanets(self, include_plantes: dict = {}):
        """
        Only compute the planets asked by the user
        :param planets: see the __init__
        :return:
        """
        include_plantes = self.__include_plantes
        include_plantes = [key for key, value in include_plantes.items() if value]
        self.__reset()

        '''Assure the 'Sun' and 'Moon' are included because they will be used again in other force models '''
        # assert 'Sun' in planets
        # assert 'Moon' in planets

        keys = self.__planets.keys()
        user_def = list(set(keys).difference(set(include_plantes)))

        if not len(user_def):
            return self

        for key in user_def:
            self.__planets.pop(key)

        return self

    def getPlanets(self):
        return self.__planets.copy()

    def __getPosByTdb(self, tdb_jd):
        """
        1. This function supports vectorized input like: time=[ 2457061.5 , 2457062.5 ]
        2. Unit: kilometers
        :param tdb_jd: TDB time in terms of JD convention
        :return: a three-dimension position vector
        """

        pos_planets = {}

        # 地球重心相对于太阳系质心的坐标
        pos1 = self.__kernel[0, 3].compute(tdb_jd)
        # 地球相对于地球重心的坐标
        pos2 = self.__kernel[3, 399].compute(tdb_jd)

        for planet in self.__planets.keys():
            index = self.__planets[planet]

            if planet != EnumType.Planets.Moon.name:
                pos_planets[planet] = (-pos1 + self.__kernel[0, index].compute(tdb_jd) - pos2) * self.__meter
            else:
                pos_planets[planet] = (self.__kernel[3, 301].compute(tdb_jd) - pos2) * self.__meter

            # print(planet, len(pos_planets[planet][0]))

        return pos_planets

    def getPosByFrame(self):
        """
        1. This function supports vectorized input like: time=[ 2457061.5 , 2457062.5 ]
        2. Unit: kilometers
        :param tdb_jd: TDB time in terms of JD convention
        :return: a three-dimension position vector
        """
        tdb_jd = self.__time.getTDB_jd
        self.__pos_planets = {}

        # 地球重心相对于太阳系质心的坐标
        pos1 = self.__kernel[0, 3].compute(tdb_jd)
        # 地球相对于地球重心的坐标
        pos2 = self.__kernel[3, 399].compute(tdb_jd)

        for planet in self.__planets.keys():
            index = self.__planets[planet]

            if planet != 'Moon':
                self.__pos_planets[planet] = (-pos1 + self.__kernel[0, index].compute(tdb_jd) - pos2) * self.__meter
            else:
                self.__pos_planets[planet] = (self.__kernel[3, 301].compute(tdb_jd) - pos2) * self.__meter

            # print(planet, len(pos_planets[planet][0]))

        return self

    def getPosPlanets(self):
        return self.__pos_planets

    def setPosVec(self, timeseries: np.array):
        """
        :param timeseries: an array of "time epochs"
        :return: an array of "positions"
        """

        self.__posVec = self.__getPosByTdb(timeseries)

        return self

    def getPosFromVec(self, index: int):
        """
        It can be used after the successful running of "setPosVec".
        :param index: the index of given time at the predefined timeseries from 'setPosVec'
        :return: position at given time epoch (expressed as 'index')
        """

        pos = {}

        for planet in self.__planets.keys():
            pos[planet] = self.__posVec[planet][:, index]

        return pos

    def getSunPosAndVel(self, time: Frame):
        """
        Used only for the relativistic force
        :param tdb_jd:
        :return: pos--meter, vel--m/s
        """
        tdb_jd = time.getTDB_jd
        pos1, vel1 = self.__kernel[0, 3].compute_and_differentiate(tdb_jd)
        pos2, vel2 = self.__kernel[3, 399].compute_and_differentiate(tdb_jd)
        pos3, vel3 = self.__kernel[0, 10].compute_and_differentiate(tdb_jd)

        return (pos3 - pos1 - pos2) * 1000, (vel3 - vel1 - vel2) / 86400 * 1000

    @property
    def printHelp(self):
        """
        2414864.50..2471184.50  Solar System Barycenter (0) -> Mercury Barycenter (1)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Venus Barycenter (2)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Earth Barycenter (3)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Mars Barycenter (4)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Jupiter Barycenter (5)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Saturn Barycenter (6)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Uranus Barycenter (7)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Neptune Barycenter (8)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Pluto Barycenter (9)
        2414864.50..2471184.50  Solar System Barycenter (0) -> Sun (10)
        2414864.50..2471184.50  Earth Barycenter (3) -> Moon (301)
        2414864.50..2471184.50  Earth Barycenter (3) -> Earth (399)
        2414864.50..2471184.50  Mercury Barycenter (1) -> Mercury (199)
        2414864.50..2471184.50  Venus Barycenter (2) -> Venus (299)
        2414864.50..2471184.50  Mars Barycenter (4) -> Mars (499)
        """
        print(self.__kernel)

        return self
