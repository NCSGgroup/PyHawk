"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/3/19 下午8:51
@Description:
"""

import numpy as np


class OrbitalElements:
    """
    This class deals with the conversion between kepler orbital elements and state vectors.
    This is a vectorized version.
    """

    def __init__(self, GM=398600.44150e9):
        self.__xyz = None
        self.__kep = None
        self.__GM = GM
        pass

    def setState(self, x, y, z, vx, vy, vz):
        """
        XYZ in GCRS
        :param x: posX [m]
        :param y: posY
        :param z: posZ
        :param vx: velX [m/s]
        :param vy: velY
        :param vz: velZ
        :return:
        """

        '''
        Reference:
        CC NAME       :  XYZELE
        CC
        CC    CALL XYZELE(GM,T,X,V,A,E,I,KN,PER,T0)
        CC
        CC PURPOSE    :  THIS SUBROUTINE CALCULATES THE (OSCULATING)
        CC               ORBITAL ELEMENTS OF A CELESTIAL BODY , WHOSE
        CC               CARTESIAN POSITION- AND VELOCITY COMPONENTS ARE
        CC               GIVEN IN THE ARRAYS X(K) , V(K) , K=1,2,3
        CC
        CC PARAMETERS :
        CC         IN :  GM     : GRAVITY-CONSTANT (GAUSSIAN CONSTANT R*8
        CC                        PLANETARY APPLICATIONS)
        CC               T      : EPOCH TO WHICH X,V (AND ELEMENTS)   R*8
        CC                        REFER (MJD)
        CC               X,V    : POSITION AMND VELOCITY OF CELESTIAL R*8(3)
        CC                        BODY AT TIME T
        CC        OUT :  A      : SEMI MAJOR AXIS                     R*8
        CC               E      : NUMERICAL EXCENTRICITY              R*8
        CC               I      : INCLINATION OF ORBITAL PLANE WITH   R*8
        CC                        RESPECT TO FUNDAMENTAL PLANE OF
        CC                        COORDINATE SYSTEM
        CC               KN     : LONGITUDE OF ASCENDING NODE         R*8
        CC               PER    : ARGUMENT OF PERIGEE/HELION          R*8
        CC               T0
        CC               M      : Mean anomaly  [rad]                 R*8
        CC               
        CC
        CC
        CC SR CALLED  :  ---
        CC
        CC REMARKS    :  ONLY ELLIPTICAL ELEMENTS ARE CALCULATED
        CC               VERSION CHECKING WHETHER ORBIT IS ELLIPTIC  
        CC
        CC AUTHOR     :  G.BEUTLER
        CC
        CC VERSION    :  3.4  (JAN 93)
        CC
        CC CREATED    :  87/11/03 12:22        LAST MODIFIED :  80-NOV-01
        CC
        CC MODIFIED   :  28-NOV-01  GB: CHECKING WHETHER ORBIT IS ELLIPTIC
        CC
        CC COPYRIGHT  :  ASTRONOMICAL INSTITUTE
        CC      1987      UNIVERSITY OF BERNE
        CC                    SWITZERLAND
        '''

        self.__xyz = np.array([x, y, z, vx, vy, vz], dtype=np.float64).transpose()

        h1 = y * vz - z * vy
        h2 = -z * vx + x * vz
        h3 = x * vy - y * vx

        lan = np.arctan2(h1, h2)
        inc = np.arctan2(np.sqrt(h1 ** 2 + h2 ** 2), h3)

        P = (h1 ** 2 + h2 ** 2 + h3 ** 2) / self.__GM
        R = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        ecv = P / R - 1.0
        esv = np.sqrt(P / self.__GM) / R * (x * vx + y * vy + z * vz)
        V1 = np.arctan2(esv, ecv)
        ecc = np.sqrt(ecv ** 2 + esv ** 2)

        if (np.array(ecc) > 1.0).any():
            raise Exception('OrbitElements: ORBIT NO ELLIPSE, PROC STOPPED')

        ck = np.cos(lan)
        sk = np.sin(lan)
        ci = np.cos(inc)
        si = np.sin(inc)

        xx1 = ck * x + sk * y
        xx2 = -ci * sk * x + ci * ck * y + si * z
        xx3 = si * sk * x - si * ck * y + ci * z
        u = np.arctan2(xx2, xx1)
        argp = u - V1
        ex = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(V1 / 2))
        a = P / (1 - ecc ** 2)
        mano = ex - ecc * np.sin(ex)

        self.__kep = np.array([a, ecc, inc, lan, argp, mano]).transpose()
        pass

    def setKepler(self, a, ecc, inc, lan, argp, mano):
        """
        Kepler orbital Elements: a,ecc define the shape; inc, lan define the attitude; argp and mano define the movement
        :param a: semi-major axis [meters]
        :param ecc: eccentricity [dimensionless]
        :param inc: inclination [rad]
        :param lan: longitude of the ascending node [rad]
        :param argp: argument of the perigee [rad]
        :param mano: mean anomaly [rad]
        :return:
        """

        '''
        RReference:
        CC NAME       :  EPHEM
        CC
        CC      CALL EPHEM(GM,A,E,I,KNOT,PERI,T0,T,X,XP)
        CC
        CC PURPOSE    :  COMPUTE POSITION X AND VELOCITY XP OF A CELESTIAL
        CC               BODY WHOSE OSCULATING ELEMENTS ARE GIVEN
        CC               (NO PARABOLAS, HYPERBOLAS)
        CC
        CC PARAMETERS :
        CC         IN :  GM     : GRAVITY - CONSTANT                  R*8
        CC               A      : SEMIMAJOR AXIS                      R*8
        CC               E      : NUMERICAL EXCENTRICITY              R*8
        CC               I      : INCLINATION (RADIAN)                R*8
        CC               KNOT   : RIGHT ASCENSION OF ASCENDING NODE   R*8
        CC                        (RADIAN)
        CC               PERI   : ARGUMENT OF PERICENTRE (RADIAN)     R*8
        CC               T0     : PERICENTRE-PASSING-TIME             R*8
        CC               T      : EPOCH OF EPHEMERIS COMPUTATION      R*8
        CC               X      : POSITION OF SATELLITE               R*8
        CC               XP     : VELOCITY OF SATELLITE               R*8
        CC               X(K), XP(K),K=1,2,3: POSITION AND VELOCITY   R*8
        CC
        CC SR CALLED  :  ---
        CC
        CC REMARKS    :  ---
        CC
        CC AUTHOR     :  G.BEUTLER, M.ROTHACHER
        CC
        CC VERSION    :  3.4  (JAN 93)
        CC
        CC CREATED    :  87/11/30 08:11        LAST MODIFIED :  13-AUG-2003
        CC
        CC MODIFIED   :  13-AUG-2003 : GB: NO ENDLESS LOOP BY KEPLER'S EQUATION
        CC
        CC COPYRIGHT  :  ASTRONOMICAL INSTITUTE
        CC      1987      UNIVERSITY OF BERNE
        CC                    SWITZERLAND
        '''

        self.__kep = np.array([a, ecc, inc, lan, argp, mano], dtype=np.float64).transpose()

        '''P=PARAMETER OF CONIC SECTION'''
        P = a * (1.0 - ecc ** 2)

        '''N = MEAN MOTION, M = MEAN ANOMALY'''
        N = np.sqrt(self.__GM / a ** 3)
        # M = N * (T - T0) + mano
        M = mano

        '''SOLVE KEPLER'S EQUATION'''
        '''EX, EX1: ECCENTRIC ANOMALY'''
        EX1 = M
        for i in range(10):
            EX = EX1 + (M + ecc * np.sin(EX1) - EX1) / (1 - ecc * np.cos(EX1))
            EX1 = EX

        '''V = TRUE ANOMALY'''
        V = 2 * np.arctan(np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(EX / 2))
        R = a * (1.0 - ecc * np.cos(EX))
        BETA = np.sqrt(self.__GM / P)
        X1 = R * np.cos(V)
        X2 = R * np.sin(V)
        XP1 = -BETA * np.sin(V)
        XP2 = BETA * (ecc + np.cos(V))

        '''SINES AND COSINES OF INCLINATION I, NODE K, PERIGEE O'''
        CK = np.cos(lan)
        SK = np.sin(lan)
        CI = np.cos(inc)
        SI = np.sin(inc)
        CO = np.cos(argp)
        SO = np.sin(argp)

        '''VECTORS P AND Q'''
        P1 = CK * CO - SK * CI * SO
        P2 = SK * CO + CK * CI * SO
        P3 = SI * SO
        Q1 = -CK * SO - SK * CI * CO
        Q2 = -SK * SO + CK * CI * CO
        Q3 = SI * CO

        '''COMPUTE POSITION AND VELOCITY'''
        x = P1 * X1 + Q1 * X2
        y = P2 * X1 + Q2 * X2
        z = P3 * X1 + Q3 * X2

        vx = P1 * XP1 + Q1 * XP2
        vy = P2 * XP1 + Q2 * XP2
        vz = P3 * XP1 + Q3 * XP2

        self.__xyz = np.array([x, y, z, vx, vy, vz], dtype=np.float64).transpose()

        pass

    def getKepler(self):
        return self.__kep

    def getState(self):
        return self.__xyz

    def getMeanMotion(self):
        """N = MEAN MOTION, unit: [s]"""
        if len(np.shape(self.__kep)) > 1:
            a = self.__kep[:, 0]
        else:
            a = self.__kep[0]
        return np.sqrt(self.__GM / a ** 3)
