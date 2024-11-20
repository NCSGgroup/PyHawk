from src.Frame.EOP import EOP
import src.Frame.SOFA as sf
import src.Preference.Pre_Constants as Pre_Constants
from src.Preference.Pre_Frame import FrameConfig
from src.Preference.EnumType import TimeFormat, CoordinateTransform
import numpy as np


class Frame:
    """
    This Class deals with the Time Frame and Spatial Coordinate
    see reference:
    1. SOFA_lib: http://www.iausofa.org/cookbooks.html
    2. IERS: IERS convention 2010
    3. This is important.  Book: Satellite Orbits: Models, Methods and Applications. Chapter 7.3
       about the transformation with FORCE in GCRS and ITRS, eq 7.71

    Notice:
    1. As TT - TDB is quite small, Consequently, TT can be used in practice in place of TDB in the expressions for the
        fundamental nutation arguments. This also applies to the non-polynomial part of Eq. (5.16) for the GCRS CIP
        coordinates. See IERS2010 page 66. But take care for the JPL planet empheric
    2. TT =  TDT
    """
    def __init__(self, eop: EOP):
        """
        :param eop: EOP class called by 'EOP' to provide earth rotation parameters
        :param CoorTrans_method: CIO wit xy series or CIO with classical angles
        :param OmegaOption: earth rotation velocity, see getOmega
        :param tdbOption: =1: TDB =TT ; =2 : TDB in a complex form, see getTDB_complex
        """
        '''Define the method of coordinate transformation'''
        self.__method = None
        self.__OmegaOption = None
        self.__tdbOption = None
        '''Reset all property'''
        self.__reset()

        '''introduce the EOP class'''
        # self.__eop = EOP().load(Setting.path.EOP).setCorrection(True)
        self.__eop = eop
        pass

    def configure(self, frameConfig:FrameConfig):
        '''Define the method of coordinate transformation'''
        self.__method = frameConfig.CoorTrans_method

        '''define the format of Earth rotation velocity'''
        assert 1 <= frameConfig.OmegaOption <= 3
        self.__OmegaOption = frameConfig.OmegaOption

        '''define the format of TDB-MJD'''
        assert 1 <= frameConfig.tdbOption <= 2
        self.__tdbOption = frameConfig.tdbOption
        return self

    def __reset(self):
        self.__gps_mjd = None
        self.__gps_second = None
        self.__utc_mjd = None
        self.__tdb_mjd = None
        self.__tdb_jd = None
        self.__interpolated_EOP = None
        self.__theta = None
        self.__tt_mjd = None
        self.__tai_mjd = None
        self.__RotationMatrix = None

        pass

    def setTime(self, time, format=TimeFormat.GPS_second):
        """
        A smarter way to set time.
        :param format: ['second']['MJD']['JD']
        :param scale: ['GPS']['TAI']['UTC']
        :param time:
        :return:
        """

        if format is TimeFormat.GPS_second:
            self.__setGPS_from_epoch2000(time)
        elif format is TimeFormat.UTC_MJD:
            self.__setUTC_mjd(time)
        elif format is TimeFormat.TAI_MJD:
            self.__setTAI_mjd(time)

        return self

    def __setGPS_from_epoch2000(self, GPS_second):
        """
        :param GPS_second:GPS seconds past 2000-01-01 12:00:00
        :return:
        """
        J2000_mjd = Pre_Constants.J2000_MJD
        tai_mjd = (GPS_second + Pre_Constants.Diff_Tai_GPS) / 86400 + J2000_mjd
        self.__setTAI_mjd(tai_mjd)

        return self

    def __setUTC_mjd(self, utc_mjd):
        DJMJD0 = Pre_Constants.DJMJD0
        self.__utc_mjd = utc_mjd
        tai1, tai2 = sf.utctai(DJMJD0, utc_mjd)
        self.__setTAI_mjd(tai2)
        return self

    def __setTAI_mjd(self, tai_mjd):
        self.__reset()
        self.__tai_mjd = tai_mjd
        self.__run()
        return self

    def __run(self):
        '''
        Called once setTai_mjd is called
        :return:
        '''
        DJMJD0 = Pre_Constants.DJMJD0
        J2000_mjd = Pre_Constants.J2000_MJD
        TAI = self.__tai_mjd
        # TAI = np.float128(self.__tai_mjd)

        self.__gps_mjd = TAI - Pre_Constants.Diff_Tai_GPS / 86400.
        self.__gps_second = (TAI - J2000_mjd) * 86400 - Pre_Constants.Diff_Tai_GPS

        UTC1, UTC2 = sf.taiutc(DJMJD0, TAI)
        self.__utc_mjd = UTC2
        # begin = Ti.time()
        eop = self.__eop.interp(UTC2)
        # print('eop:Cost time: %s ms' % ((Ti.time() - begin) * 1000))

        # '''Polar motion (arcsec->radians).'''
        # as2r = GeoMathKit.as2rad()
        # eop['xp'] *= as2r
        # eop['yp'] *= as2r
        # ''' CIP offsets wrt IAU 2006/2000A (as->radians).'''
        # eop['dX'] *= as2r
        # eop['dY'] *= as2r
        self.__interpolated_EOP = eop

        xp, yp, dut1, dX, dY = eop['xp'], eop['yp'], eop['dut1'], eop['dX'], eop['dY']

        '''TT (MJD)'''
        TT = TAI + Pre_Constants.Diff_Tai_TT / 86400
        self.__tt_mjd = TT

        '''*TDB (MJD)'''
        if self.__tdbOption == 1:
            self.__tdb_mjd = TT
        elif self.__tdbOption == 2:
            self.__tdb_mjd = self.__getTDB_mjd_complex(TT)

        '''TDB(JD)'''
        self.__tdb_jd = self.__tdb_mjd + DJMJD0

        ''' UT1 (MJD)'''
        # dut1 = -4.451433467983630377e-01
        UT1 = UTC2 + dut1 / 86400.

        '''GMST [radian]'''
        # self.__theta = sf.gmst06(DJMJD0, UT1, DJMJD0, TT)
        Tu0 = (np.floor(self.__utc_mjd) - Pre_Constants.J2000_MJD) / 36525.0
        gmst0 = (6.0 / 24 + 41.0 / (24 * 60) + 50.54841 / (24 * 60 * 60))
        gmst0 = gmst0 + (8640184.812866 / (24 * 60 * 60)) * Tu0
        gmst0 = gmst0 + (0.093104 / (24 * 60 * 60)) * Tu0 * Tu0
        gmst0 = gmst0 + (-6.2e-6 / (24 * 60 * 60)) * Tu0 * Tu0 * Tu0
        r = 1.002737909350795 + 5.9006e-11 * Tu0 - 5.9e-15 * Tu0 * Tu0
        gmst = np.mod(2 * np.pi * (gmst0 + r * np.mod(self.__utc_mjd, 1)), 2 * np.pi)
        self.__theta = gmst

        '''CIP and CIO, IAU 2006/2000A.'''
        X, Y, S = None, None, None
        if CoordinateTransform[self.__method] == CoordinateTransform.CIOxy:
            X, Y = sf.xy06(DJMJD0, TT)
            S = sf.s06(DJMJD0, TT, X, Y)
        elif CoordinateTransform[self.__method] == CoordinateTransform.CIOangle:
            X, Y, S = sf.xys06a(DJMJD0, TT)

        '''* Add CIP corrections. '''
        X = X + dX
        Y = Y + dY
        '''* GCRS to CIRS matrix. '''
        RC2I = sf.c2ixys(X, Y, S)  # NPB matrix
        '''* Earth rotation angle. '''
        ERA = sf.era00(DJMJD0, UT1)
        '''* Form celestial-terrestrial matrix (no polar motion yet). '''
        RC2TI = sf.cr(RC2I)
        PRC2TI = sf.rz(ERA, RC2TI)  # Earth rotation matrix
        VRC2TI = self.__derivative_of_rotation_matrix(ERA, RC2TI)  # derivative of the rotation matrix

        '''* Polar motion matrix (TIRS->ITRS, IERS 2003). '''
        RPOM = sf.pom00(xp, yp, sf.sp00(DJMJD0, TT))

        '''* Form celestial-terrestrial matrix (including polar motion). '''
        PRC2IT = sf.rxr(RPOM, PRC2TI)  # polar motion matrix
        VRC2IT = sf.rxr(RPOM, VRC2TI)

        self.__RotationMatrix = (np.array(PRC2IT), np.array(VRC2IT))
        pass

    def __getTDB_mjd_complex(self, Mjd_TT):
        """
        Computes the Modified Julian Date for barycentric dynamical time

        Reference:
        % Vallado D. A; Fundamentals of Astrodynamics and Applications; McGraw-Hill;
        % New York; 3rd edition(2007).

        :param Mjd_TT: TT in MJD
        :return: TDB in MJD
        """
        T_TT = (Mjd_TT - Pre_Constants.J2000_MJD) / 36525

        Mjd_TDB = Mjd_TT + (0.001658 * np.sin(628.3076 * T_TT + 6.2401)
                            + 0.000022 * np.sin(575.3385 * T_TT + 4.2970)
                            + 0.000014 * np.sin(1256.6152 * T_TT + 6.1969)
                            + 0.000005 * np.sin(606.9777 * T_TT + 4.0212)
                            + 0.000005 * np.sin(52.9691 * T_TT + 0.4444)
                            + 0.000002 * np.sin(21.3299 * T_TT + 5.5431)
                            + 0.000010 * np.sin(628.3076 * T_TT + 4.2490)) / 86400

        return Mjd_TDB

    def __derivative_of_rotation_matrix(self, ERA, R):
        """
        This is used to compute the derivative of rotation matrix for velocity.

        r = P*r
        v = dr/dt = dP/dt * r + P * v
        P = PNB * EarthRotation(M) * PolarMotion
        dP/dt = PNB * dM/dt * PolarMotion

        M = Rtheta
        Rtheta =  (  + cos(psi)   + sin(psi)     0  )
                  (                                 )
                  (  - sin(psi)   + cos(psi)     0  )
                  (                                 )
                  (       0            0         1  )

        dM/dt = [ 0   1   0 ]
                [ -1  0   0 ]  * Rtheta * dpsi/dt
                [0    0   0]

        dpsi/dt = 1.00273781191135448 rev/day: the Earth rotation velocity, refer to IERS2010 Chapter 1 Tab 1.1 page 18

        :return:
        """

        '''Earth rotation angle velocity [rad/s]'''
        Omega = self.getOmega

        S = np.sin(ERA)
        C = np.cos(ERA)

        M = np.array([
            [-S, C, 0],
            [-C, -S, 0],
            [0, 0, 0]
        ])

        return np.dot(Omega * M, np.array(R))

    @property
    def getGPS_mjd(self):
        return self.__gps_mjd

    @property
    def getGPS_jd(self):
        return self.__gps_mjd + Pre_Constants.DJMJD0

    @property
    def getTDB_mjd(self):
        return self.__tdb_mjd

    @property
    def getTDB_jd(self):
        return self.__tdb_jd

    @property
    def getTT_mjd(self):
        return self.__tt_mjd

    @property
    def getUTC_mjd(self):
        return self.__utc_mjd

    @property
    def getGPS_from_epoch2000(self):
        return self.__gps_second

    @property
    def getTheta(self):
        """

        :return: GMST in [radian]
        """
        return self.__theta

    @property
    def getOmega(self):
        """
        Option 1:  1.00273781191135448 rev/day: the Earth rotation velocity, refer to IERS2010 Chapter 1 Tab 1.1 page 18
        Option 2 and Option 3 refer to
        https://ww2.mathworks.cn/matlabcentral/fileexchange/55167-high-precision-orbit-propagator

        :return: the Earth self-rotation velocity [rad/s]
        """
        Omega = None

        # option 1
        if self.__OmegaOption == 1:
            '''derivative of GMST at J2000'''
            Omega = 1.00273781191135448 * np.pi * 2 / 86400
        # option 2
        elif self.__OmegaOption == 2:
            Omega = 1.00273781191135448 * np.pi * 2 / 86400 - 0.843994809e-9 * self.getEOP['LOD']
        # option 3
        elif self.__OmegaOption == 3:
            Omega = 7292115.8553e-11 + 4.3e-15 * ((self.getUTC_mjd - Pre_Constants.J2000_MJD) / 36525)

        return Omega

    @property
    def getEOP(self):
        """
        Get EOP parameters at given time
        :return:
        self.__EOP = {
            'xp': xp,  [radian]
            'yp': yp,[radian]
            'dut1': dut1,[second]
            'dX': dX,[radian]
            'dY': dY[radian]
            'LOD': LOD [second]
        }
        """
        return self.__interpolated_EOP.copy()

    @property
    def getRotationMatrix(self):
        """
        Get rotation matrix P from celestial (GCRS) to terrestrial (ITRS) matrix
        And its derivative dP/dt
        :return: a tuple of (P, dP/dt)
        """

        return self.__RotationMatrix

    @staticmethod
    def AccGCRS2ITRS(acc_GCRS: np.ndarray, RotationMatrix):
        """
        Transform Acc from GCRS to ITRS.
        NOT limited to Acc. For example, this is also available for da/dc da/ds
        :param acc_GCRS: acceleration in GCRS frame
        :param RotationMatrix: [0] RC2IT for pos   [1] RC2IT for velocity
        :return: a matrix of dimension [3, -1]
        """
        assert np.shape(acc_GCRS)[0] == 3

        rp = RotationMatrix[0]
        acc_ITRS = np.dot(rp, acc_GCRS.reshape((3, -1)))

        return acc_ITRS

    @staticmethod
    def AccITRS2GCRS(acc_ITRS: np.ndarray, RotationMatrix):
        """
        Transform Acc from GCRS to ITRS.
        NOT limited to Acc. For example, this is also available for da/dc da/ds
        :param acc_ITRS:
        :param RotationMatrix:
        :return: a matrix in dimension [3, -1]
        """
        assert np.shape(acc_ITRS)[0] == 3

        rp = RotationMatrix[0]
        acc_GCRS = np.dot(np.transpose(rp), acc_ITRS.reshape((3, -1)))
        return acc_GCRS

    @staticmethod
    def PosGCRS2ITRS(pos_GCRS: np.ndarray, RotationMatrix):
        """
        Transform position vector from GCRS to ITRS
        :param pos_GCRS: a vector
        :param RotationMatrix:
        :return: a vector
        """
        rp = RotationMatrix[0]
        pos_ITRS = np.dot(rp, pos_GCRS.reshape((3, 1)))

        return pos_ITRS.flatten()

    @staticmethod
    def PosITRS2GCRS(pos_ITRS: np.ndarray, RotationMatrix):
        """
        Transform position vector from ITRS to GCRS
        :param pos_ITRS:
        :param RotationMatrix:
        :return: a vector
        """
        rp = RotationMatrix[0]
        pos_GCRS = np.dot(np.transpose(rp), pos_ITRS.reshape((3, 1)))

        return pos_GCRS.flatten()

    @staticmethod
    def VelGCRS2ITRS(pos_GCRS: np.ndarray, vel_GCRS: np.ndarray, RotationMatrix):
        """
        Transform the velocity vector from GCRS to ITRS
        :param pos_GCRS:
        :param vel_GCRS:
        :param RotationMatrix:
        :return: a vector
        """
        rp, vp = RotationMatrix[0], RotationMatrix[1]

        vel_GCRS = np.dot(rp, vel_GCRS.reshape((3, 1))) + np.dot(vp, pos_GCRS.reshape((3, 1)))

        return vel_GCRS.flatten()

    @staticmethod
    def VelITRS2GCRS(pos_ITRS: np.ndarray, vel_ITRS: np.ndarray, RotationMatrix):
        """
        Transform the velocity vector from ITRS to GCRS
        :param pos_ITRS:
        :param vel_ITRS:
        :param RotationMatrix:
        :return: a vector
        """
        rp, vp = RotationMatrix[0], RotationMatrix[1]

        vel_ITRS = np.dot(np.transpose(rp), vel_ITRS.reshape((3, 1))) + \
                   np.dot(np.transpose(vp), pos_ITRS.reshape((3, 1)))

        return vel_ITRS.flatten()

    @staticmethod
    def dadrITRS2GCRS(dadr_ITRS: np.ndarray, RotationMatrix):
        """
        Transform da/dr from ITRS to GCRS, notice the format of da/dr should be
        da_x/dx ...
        da_y/dx ...
        da_z/dx ...

        :param dadr_ITRS: dimension [3,3]
        :param RotationMatrix:
        :return:  matrix in dimension [3,3]
        """

        assert np.shape(dadr_ITRS) == (3, 3)
        rp = RotationMatrix[0]

        dadr_GCRS = np.dot(np.transpose(rp), dadr_ITRS)
        dadr_GCRS = np.dot(dadr_GCRS, rp)

        return dadr_GCRS

    @staticmethod
    def dadrGCRS2ITRS(dadr_GCRS: np.ndarray, RotationMatrix):
        """
        Transform da/dr from GCRS to ITRS, notice the format of da/dr should be
        da_x/dx ...
        da_y/dx ...
        da_z/dx ...

        :param dadr_GCRS:
        :param RotationMatrix:
        :return: matrix in dimension [3,3]
        """
        assert np.shape(dadr_GCRS) == (3, 3)
        rp = RotationMatrix[0]
        dadr_ITRS = np.dot(rp, dadr_GCRS)
        dadr_ITRS = np.dot(dadr_ITRS, np.transpose(rp))

        return dadr_ITRS

    @staticmethod
    def mjd2cal(mjd):
        """
        Convert mjd to calender
        :param mjd:
        :return:
        """
        DJMJD0 = 2400000.5
        IY, IM, ID, FD = sf.jd2cal(DJMJD0, mjd)
        IH = (FD * 86400) // 3600
        Min = (FD * 86400 - IH * 3600) // 60
        Sec = FD * 86400 - IH * 3600 - Min * 60

        return IY, IM, ID, IH, Min, Sec

    @staticmethod
    def mjd2sec(mjd):
        '''seconds past epoch 2000-01-01 12:00:00'''
        J2000_mjd = 51544.5
        return (mjd - J2000_mjd) * 86400

    @staticmethod
    def cal2mjd(IY, IM, ID, IH, MIN, SEC):
        """
        Convert calender to mjd
        :param IY:
        :param IM:
        :param ID:
        :param IH:
        :param MIN:
        :param SEC:
        :return:
        """

        DJMJD0, DATE = sf.cal2jd(IY, IM, ID)

        TIME = (60 * (60 * IH + MIN) + SEC) / 86400
        mjd = DATE + TIME

        return mjd

