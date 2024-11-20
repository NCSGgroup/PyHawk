"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2022/8/8
@Description:
"""
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Preference.Pre_Frame import FrameConfig
import numpy as np


class EOP:
    """
    This is a function to provide interpolated "Earth Orientation Parameters" at given epoch.
    Validation has not yet been done with the previous FORTRAN code.
    Steps to use the class should follow as below:
    1. download the EOP files from IERS website
    2. gather all the relevant files in a directory defined by yourself, and make sure the path can be correctly read by
    the "load" function
    3. run "interp" function at given epoch to output a tuple composed of X, Y and UT1-UTC

    Notice: the input should be UTC time in MJD convention

    reference:
    1. IERS2010 chapter 5, section 5.5.1 and 5.5.3 ,page 50
    """

    def __init__(self):
        """

        :param isCorrection: To decide if the correction for EOP parameters due to oceanic effect and Lunisolar effect is carried out or not.
        :param isRadian: the unit will be switched from [Arcsecond] to [Radian]
        """
        self.__radian = GeoMathKit.as2rad()
        self.__isCorrection = None
        self.__frameConfig = None
        self.__framePath = None
        self.__pre_save_data()
        pass

    def configure(self, frameConfig: FrameConfig):
        self.__frameConfig = frameConfig
        self.__isCorrection = self.__frameConfig.isCorrection_EOP
        if not self.__frameConfig.isRadian_EOP:
            self.__radian = 1
        return self

    def __pre_save_data(self):
        """
        this is purely for saving memory and accelerating the computation.
        :return:
        """
        self.__ocean_narg = np.zeros((71, 6), np.int64)
        self.__ocean_xsin = np.zeros(71)
        self.__ocean_xcos = np.zeros(71)
        self.__ocean_ysin = np.zeros(71)
        self.__ocean_ycos = np.zeros(71)
        self.__ocean_utsin = np.zeros(71)
        self.__ocean_utcos = np.zeros(71)

        self.__grav_narg = np.zeros((10, 6), np.int64)
        self.__grav_xsin = np.zeros(10)
        self.__grav_xcos = np.zeros(10)
        self.__grav_ysin = np.zeros(10)
        self.__grav_ycos = np.zeros(10)

        self.__ocean_ARG_coef = np.array([
            [648000.0 + 67310.54841 * 15.0, (876600 * 3600 + 8640184.812866) * 15.0, 0.093104 * 15.0, - 6.2e-6 * 15.0,
             0],
            [485868.249036, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
            [1287104.79305, 129596581.0481, - 0.5532, - 0.000136, -0.00001149],
            [335779.526232, 1739527262.8478, - 12.7512, - 0.001037, 0.00000417],
            [1072260.70369, 1602961601.2090, - 6.3706, 0.006593, -0.00003169],
            [450160.398036, - 6962890.2665, 7.4722, 0.007702, -0.00005939]
        ])

        self.__ocean_DARG_coef = np.array([
            [(876600 * 3600 + 8640184.812866) * 15.0, 2. * 0.093104 * 15.0, 3. * 6.2e-6 * 15.0, 0],
            [1717915923.2178, 2. * 31.8792, 3. * 0.051635, -4. * 0.00024470],
            [129596581.0481, - 2. * 0.5532, - 3. * 0.000136, -4. * 0.00001149],
            [1739527262.8478, - 2. * 12.7512, - 3. * 0.001037, 4. * 0.00000417],
            [1602961601.2090, - 2. * 6.3706, 3. * 0.006593, -4. * 0.00003169],
            [- 6962890.2665, 2. * 7.4722, 3. * 0.007702, -4. * 0.00005939]
        ])

        self.__grav_ARG_coef = np.array([
            [67310.54841 * 15.0 + 648000.0, (876600 * 3600 + 8640184.812866) * 15.0, 0.093104 * 15.0, 6.2e-6 * 15.0, 0],
            [485868.249036, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
            [1287104.79305, 129596581.0481, - 0.5532, - 0.000136, -0.00001149],
            [335779.526232, 1739527262.8478, - 12.7512, - 0.001037, 0.00000417],
            [1072260.70369, 1602961601.2090, - 6.3706, 0.006593, -0.00003169],
            [450160.398036, - 6962890.2665, 7.4722, 0.007702, -0.00005939]
        ])

        pass

    def load(self, **kwargs):
        """
        load relevant files for the EOP computation
        :param EOP_path:
        :return:
        """
        self.__framePath = self.__frameConfig.PathOfFiles()
        self.__framePath.__dict__.update(self.__frameConfig.PathOfFilesConfig.copy())
        '''load EOP file'''
        with open(self.__framePath.EOP) as f:
            content = f.readlines()
            pass

        self.__polarX = []
        self.__polarY = []
        self.__UT1 = []
        self.__RJD = []
        self.__dX = []
        self.__dY = []
        self.__LOD = []

        # The starting lines are not required
        for elem in content[14:]:
            value = elem.split()
            self.__RJD.append(value[3])
            self.__polarX.append(value[4])
            self.__polarY.append(value[5])
            self.__UT1.append(value[6])
            self.__LOD.append(value[7])
            self.__dX.append(value[8])
            self.__dY.append(value[9])

        self.__RJD = np.array(self.__RJD).astype(np.int64)
        self.__polarX = np.array(self.__polarX).astype(np.float64)
        self.__polarY = np.array(self.__polarY).astype(np.float64)
        self.__UT1 = np.array(self.__UT1).astype(np.float64)
        self.__LOD = np.array(self.__LOD).astype(np.float64)
        self.__dX = np.array(self.__dX).astype(np.float64)
        self.__dY = np.array(self.__dY).astype(np.float64)

        '''load file of ocean correction'''
        if self.__isCorrection:
            ocean_correction_path = self.__framePath.EOPocean
            grav_correction_path = self.__framePath.EOPgrav
            self.__loadEOPocean(ocean_correction_path)
            self.__loadEOPgrav(grav_correction_path)

        return self

    def __loadEOPocean(self, dirEop):

        with open(dirEop) as f:
            content = f.readlines()

        assert len(content) == len(self.__ocean_utcos)

        for i in range(len(content)):
            value = content[i].split("&")[1].split(",")
            self.__ocean_narg[i, :] = value[0:6]
            self.__ocean_xsin[i] = value[6]
            self.__ocean_xcos[i] = value[7]
            self.__ocean_ysin[i] = value[8]
            self.__ocean_ycos[i] = value[9]
            self.__ocean_utsin[i] = value[10]
            self.__ocean_utcos[i] = value[11]

        pass

    def __loadEOPgrav(self, dirEop):

        with open(dirEop) as f:
            content = f.readlines()

        assert len(content) == len(self.__grav_xcos)

        for i in range(len(content)):
            value = content[i].split("&")[1].split(",")
            self.__grav_narg[i, :] = value[0:6]
            self.__grav_xsin[i] = value[6]
            self.__grav_xcos[i] = value[7]
            self.__grav_ysin[i] = value[8]
            self.__grav_ycos[i] = value[9]

        pass

    def interp(self, utc_mjd_int):
        """
        Notice: make sure the interpolated epoch is restricted within the range of EOP file
         This subroutine takes a series of x, y, and UT1-UTC values
         and interpolates them to an epoch of choice. This routine
          assumes that the values of x and y are in seconds of
          arc and that UT1-UTC is in seconds of time. At least
          one point before and one point after the epoch of the
           interpolation point are necessary in order for the
           interpolation scheme to work.
        :param utc_mjd_int:epoch for the interpolated value, UTC in MJD format
        :return:
         xp   - interpolated value of x (polar motion), [arc second]
         yp   - interpolated value of y (polar motion)  [arc second]
         dut1 - interpolated value of ut1-utc  [second]
         dX -... [arc second]
         dY -... [arc second]

         Modified by Yang from original codes of Ch. BIZOUARD (Observatoire de Paris)
        """
        assert self.__RJD[0] <= utc_mjd_int <= self.__RJD[-1]

        index0 = int(utc_mjd_int) - self.__RJD[0]
        dim = len(self.__RJD)

        if index0 <= 1:
            index = [0, 1, 2, 3]
        elif index0 >= dim - 3:
            index = [dim - 4, dim - 3, dim - 2, dim - 1]
        else:
            index = [index0 - 1, index0, index0 + 1, index0 + 2]

        # begin = Ti.time()
        xp = GeoMathKit.Lagint4(self.__RJD, self.__polarX, utc_mjd_int, index)
        yp = GeoMathKit.Lagint4(self.__RJD, self.__polarY, utc_mjd_int, index)
        dut1 = GeoMathKit.Lagint4(self.__RJD, self.__UT1, utc_mjd_int, index)
        LOD = GeoMathKit.Lagint4(self.__RJD, self.__LOD, utc_mjd_int, index)
        dX = GeoMathKit.Lagint4(self.__RJD, self.__dX, utc_mjd_int, index)
        dY = GeoMathKit.Lagint4(self.__RJD, self.__dY, utc_mjd_int, index)
        # print('lagin4-1:Cost time: %s ms' % ((Ti.time() - begin) * 1000))

        # begin = Ti.time()
        # xpl = GeoMathKit.lagint4(self.__RJD, self.__polarX, utc_mjd_int)
        # ypl = GeoMathKit.lagint4(self.__RJD, self.__polarY, utc_mjd_int)
        # dut1l = GeoMathKit.lagint4(self.__RJD, self.__UT1, utc_mjd_int)
        # LODl = GeoMathKit.lagint4(self.__RJD, self.__LOD, utc_mjd_int)
        # dXl = GeoMathKit.lagint4(self.__RJD, self.__dX, utc_mjd_int)
        # dYl = GeoMathKit.lagint4(self.__RJD, self.__dY, utc_mjd_int)
        # print('lagin4:Cost time: %s ms' % ((Ti.time() - begin) * 1000))

        if not self.__isCorrection:
            return {'xp': xp * self.__radian, 'yp': yp * self.__radian, 'dut1': dut1,
                    'dX': dX * self.__radian, 'dY': dY * self.__radian, 'LOD': LOD}

        # oceanic effect
        # cor_x, cor_y, cor_ut1, cor_lod = self.__PMUT1_OCEANS(utc_mjd_int)
        cor_x, cor_y, cor_ut1, cor_lod = self.__ocean(utc_mjd_int)

        xp = xp + cor_x
        yp = yp + cor_y
        dut1 = dut1 + cor_ut1

        # Lunisolar effect
        # cor_x, cor_y = self.__PM_GRAVI(utc_mjd_int)
        cor_x, cor_y = self.__grav(utc_mjd_int)
        xp = xp + cor_x
        yp = yp + cor_y

        return {'xp': xp * self.__radian, 'yp': yp * self.__radian, 'dut1': dut1,
                'dX': dX * self.__radian, 'dY': dY * self.__radian, 'LOD': LOD}

    def __ocean(self, utc_mjd):
        """
        This subroutine provides, in time domain, the diurnal/subdiurnal
            !   tidal effets on polar motion ("), UT1 (s) and LOD (s). The tidal terms,
            !   listed in the program above, have been extracted from the procedure
            !   ortho_eop.f coed by Eanes in 1997.
            !
            !   N.B.:  The fundamental lunisolar arguments are those of Simon et al.
            !
            !   These corrections should be added to "average"
            !   EOP values to get estimates of the instantaneous values.
            !
            !    PARAMETERS ARE :
            !    rjd      - epoch of interest given in mjd
            !    cor_x    - tidal correction in x (sec. of arc)
            !    cor_y    - tidal correction in y (sec. of arc)
            !    cor_ut1  - tidal correction in UT1-UTC (sec. of time)
            !    cor_lod  - tidal correction in length of day (sec. of time)
            !
            !    coded by Ch. Bizouard (2002), initially coded by McCarthy and
            !    D.Gambis(1997) for the 8 prominent tidal waves.
        :param utc_mjd: rjd
        :return:  this is a matrix-based '__PMUT1_OCEANS'
        """
        halfpi, secrad = np.pi / 2, np.pi / (180 * 3600)

        T = (utc_mjd - 51544.5) / 36525.0

        timeArr = np.array([1, T, T ** 2, T ** 3, T ** 4])
        arg = np.dot(self.__ocean_ARG_coef, timeArr)
        darg = np.dot(self.__ocean_DARG_coef, timeArr[0:-1])

        ARG = np.mod(arg, 1296000.) * secrad
        DARG = darg * secrad / 36525.

        # ------------CORRECTIONS---------------------
        ag = np.dot(self.__ocean_narg, ARG)
        dag = np.dot(self.__ocean_narg, DARG)
        ag = np.mod(ag, 4 * halfpi)

        v1 = self.__ocean_xcos * np.cos(ag) + self.__ocean_xsin * np.sin(ag)
        v2 = self.__ocean_ycos * np.cos(ag) + self.__ocean_ysin * np.sin(ag)
        v3 = self.__ocean_utcos * np.cos(ag) + self.__ocean_utsin * np.sin(ag)
        v4 = v3 * dag

        cor_x, cor_y, cor_ut1, cor_lod = v1.sum() * 1e-6, v2.sum() * 1e-6, \
                                         v3.sum() * 1e-6, v4.sum() * 1e-6

        '''
        Unit
        cor_x  arcsecond(")
        cor_y  arcsecond(")
        cor_ut1 seconds(s)
        cor_load seconds(s)
        '''
        return cor_x, cor_y, cor_ut1, cor_lod

    def __grav(self, utc_mjd):
        """
        !   This subroutine provides, in time domain, the diurnal
        !   lunisolar effet on polar motion(")
        !
        !   N.B.: The fundamental lunisolar arguments are those of Simon et al.
        !
        !   These corrections should be added to "average"
        !   EOP values to get estimates of the instantaneous values.
        !
        !    PARAMETERS ARE:
        !    rjd - epoch of interest given in mjd
        !    cor_x - tidal correction in x(sec.of arc)
        !    cor_y - tidal correction in y(sec.of arc)
        !
        !    coded by Ch.Bizouard(2002)
        :param rjd:
        :return:
        """

        halfpi, secrad = np.pi / 2, np.pi / (180 * 3600)
        T = (utc_mjd - 51544.5) / 36525.0

        timeArr = np.array([1, T, T ** 2, T ** 3, T ** 4])
        arg = np.dot(self.__grav_ARG_coef, timeArr)
        ARG = np.mod(arg, 1296000.) * secrad

        # Corrections
        ag = np.dot(self.__grav_narg, ARG)
        ag = np.mod(ag, 4 * halfpi)

        v1 = self.__grav_xcos * np.cos(ag) + self.__grav_xsin * np.sin(ag)
        v2 = self.__grav_ycos * np.cos(ag) + self.__grav_ysin * np.sin(ag)

        cor_x, cor_y = v1.sum() * 1e-6, v2.sum() * 1e-6

        return cor_x, cor_y

    @DeprecationWarning
    def __PMUT1_OCEANS(self, rjd):
        """
        This subroutine provides, in time domain, the diurnal/subdiurnal
            !   tidal effets on polar motion ("), UT1 (s) and LOD (s). The tidal terms,
            !   listed in the program above, have been extracted from the procedure
            !   ortho_eop.f coed by Eanes in 1997.
            !
            !   N.B.:  The fundamental lunisolar arguments are those of Simon et al.
            !
            !   These corrections should be added to "average"
            !   EOP values to get estimates of the instantaneous values.
            !
            !    PARAMETERS ARE :
            !    rjd      - epoch of interest given in mjd
            !    cor_x    - tidal correction in x (sec. of arc)
            !    cor_y    - tidal correction in y (sec. of arc)
            !    cor_ut1  - tidal correction in UT1-UTC (sec. of time)
            !    cor_lod  - tidal correction in length of day (sec. of time)
            !
            !    coded by Ch. Bizouard (2002), initially coded by McCarthy and
            !    D.Gambis(1997) for the 8 prominent tidal waves.
        :param rjd:
        :return:
        """

        halfpi, secrad = np.pi / 2, np.pi / (180 * 3600)

        T = (rjd - 51544.5) / 36525.0

        ARG, DARG = np.zeros(6), np.zeros(6)

        # ---------------------------------------------------------
        ARG[0] = (67310.54841 + (876600 * 3600 + 8640184.812866) * T
                  + 0.093104 * T ** 2 - 6.2e-6 * T ** 3) * 15.0 + 648000.0

        # ARG[0] = (ARG[0] % 1296000) * secrad

        DARG[0] = (876600 * 3600 + 8640184.812866
                   + 2. * 0.093104 * T - 3. * 6.2e-6 * T ** 2) * 15.

        # DARG[0] = DARG[0] * secrad / 36525.0  # rad / day

        ARG[1] = -0.00024470 * T ** 4 + 0.051635 * T ** 3 + 31.8792 * T ** 2 \
                 + 1717915923.2178 * T + 485868.249036

        # ARG[1] = (ARG[1] % 1296000) * secrad

        DARG[1] = -4. * 0.00024470 * T ** 3 + 3. * 0.051635 * T ** 2 \
                  + 2. * 31.8792 * T + 1717915923.2178

        # DARG[1] = DARG[1] * secrad / 36525.0  # rad / day

        ARG[2] = -0.00001149 * T ** 4 - 0.000136 * T ** 3 - 0.5532 * T ** 2 \
                 + 129596581.0481 * T + 1287104.79305

        # ARG[2] = (ARG[2] % 1296000) * secrad

        DARG[2] = -4. * 0.00001149 * T ** 3 - 3. * 0.000136 * T ** 2 \
                  - 2. * 0.5532 * T + 129596581.0481

        # DARG[2] = DARG[2] * secrad / 36525.0  # rad / day

        ARG[3] = 0.00000417 * T ** 4 - 0.001037 * T ** 3 - 12.7512 * T ** 2 \
                 + 1739527262.8478 * T + 335779.526232

        DARG[3] = 4. * 0.00000417 * T ** 3 - 3. * 0.001037 * T ** 2 \
                  - 2. * 12.7512 * T + 1739527262.8478

        ARG[4] = -0.00003169 * T ** 4 + 0.006593 * T ** 3 - 6.3706 * T ** 2 \
                 + 1602961601.2090 * T + 1072260.70369

        DARG[4] = -4. * 0.00003169 * T ** 3 + 3. * 0.006593 * T ** 2 \
                  - 2. * 6.3706 * T + 1602961601.2090

        ARG[5] = -0.00005939 * T ** 4 + 0.007702 * T ** 3 \
                 + 7.4722 * T ** 2 - 6962890.2665 * T + 450160.398036

        DARG[5] = -4. * 0.00005939 * T ** 3 + 3. * 0.007702 * T ** 2 \
                  + 2. * 7.4722 * T - 6962890.2665

        ARG = np.mod(ARG, 1296000.) * secrad
        DARG = DARG * secrad / 36525.

        # ------------CORRECTIONS---------------------

        ag, dag = np.zeros(len(self.__ocean_utcos)), np.zeros(len(self.__ocean_utcos))

        for j in range(len(self.__ocean_utcos)):
            temp1 = ARG * self.__ocean_narg[j]
            temp2 = DARG * self.__ocean_narg[j]

            ag[j] = temp1.sum()
            dag[j] = temp2.sum()
            ag[j] = np.mod(ag[j], 4 * halfpi)

        v1 = self.__ocean_xcos * np.cos(ag) + self.__ocean_xsin * np.sin(ag)
        v2 = self.__ocean_ycos * np.cos(ag) + self.__ocean_ysin * np.sin(ag)
        v3 = self.__ocean_utcos * np.cos(ag) + self.__ocean_utsin * np.sin(ag)
        v4 = v3 * dag

        cor_x, cor_y, cor_ut1, cor_lod = v1.sum() * 1e-6, v2.sum() * 1e-6, \
                                         v3.sum() * 1e-6, v4.sum() * 1e-6

        '''
        Unit
        cor_x  arcsecond(")
        cor_y  arcsecond(")
        cor_ut1 seconds(s)
        cor_load seconds(s)
        '''
        return cor_x, cor_y, cor_ut1, cor_lod

    @DeprecationWarning
    def __PM_GRAVI(self, rjd):
        """
        !   This subroutine provides, in time domain, the diurnal
        !   lunisolar effet on polar motion(")
        !
        !   N.B.: The fundamental lunisolar arguments are those of Simon et al.
        !
        !   These corrections should be added to "average"
        !   EOP values to get estimates of the instantaneous values.
        !
        !    PARAMETERS ARE:
        !    rjd - epoch of interest given in mjd
        !    cor_x - tidal correction in x(sec.of arc)
        !    cor_y - tidal correction in y(sec.of arc)
        !
        !    coded by Ch.Bizouard(2002)
        :param rjd:
        :return:
        """

        halfpi, secrad = np.pi / 2, np.pi / (180 * 3600)
        T = (rjd - 51544.5) / 36525.0

        ARG = np.zeros(6)

        ARG[0] = (67310.54841 + (876600 * 3600 + 8640184.812866) * T +
                  0.093104 * T ** 2 - 6.2e-6 * T ** 3) * 15.0 + 648000.0

        ARG[1] = -0.00024470 * T ** 4 + 0.051635 * T ** 3 + 31.8792 * T ** 2 \
                 + 1717915923.2178 * T + 485868.249036

        ARG[2] = -0.00001149 * T ** 4 - 0.000136 * T ** 3 - 0.5532 * T ** 2 \
                 + 129596581.0481 * T + 1287104.79305

        ARG[3] = 0.00000417 * T ** 4 - 0.001037 * T ** 3 - 12.7512 * T ** 2 \
                 + 1739527262.8478 * T + 335779.526232

        ARG[4] = -0.00003169 * T ** 4 + 0.006593 * T ** 3 - 6.3706 * T ** 2 \
                 + 1602961601.2090 * T + 1072260.70369

        ARG[5] = -0.00005939 * T ** 4 + 0.007702 * T ** 3 + 7.4722 * T ** 2 \
                 - 6962890.2665 * T + 450160.398036

        ARG = np.mod(ARG, 1296000.) * secrad

        # Corrections
        ag = np.zeros(len(self.__grav_xcos))

        for j in range(len(self.__grav_xcos)):
            temp1 = ARG * self.__grav_narg[j]

            ag[j] = temp1.sum()
            ag[j] = np.mod(ag[j], 4 * halfpi)

        v1 = self.__grav_xcos * np.cos(ag) + self.__grav_xsin * np.sin(ag)
        v2 = self.__grav_ycos * np.cos(ag) + self.__grav_ysin * np.sin(ag)

        cor_x, cor_y = v1.sum() * 1e-6, v2.sum() * 1e-6

        return cor_x, cor_y