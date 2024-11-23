import numpy as np
from Quaternion import Quat, Quaternion
from src.Preference.EnumType import Payload
import h5py
import pathlib as pathlib
import src.Preference.Pre_Constants as Pre_Constants
from src.Interface.ArcOperation import ArcData
from src.Preference.Pre_Accelerometer import AccelerometerConfig
import os


class AccCaliPar:
    def __init__(self, sat, arcNo, accConfig:AccelerometerConfig,**kwargs):
        self.sat = sat
        self.arcNo = arcNo
        self.accConfig = accConfig
        # if kwargs:
        #     self.__parFlag = 0
        #     self.AccCaliValue = kwargs.get('AccCaliPar')
        # else:
        #     self.__parFlag = 1
        self.__par = self.__configure()

    def __configure(self):
        cali_value = None
        # if self.__parFlag == 0:
        #     cali_value = self.AccCaliValue
        #======= strategy ======
        isScale = self.accConfig.cali['Scale'].copy()
        isBias_constant = self.accConfig.cali['Bias_constant'].copy()
        isBias_trend = self.accConfig.cali['Bias_trend'].copy()
        isBias_quadratic = self.accConfig.cali['Bias_quadratic'].copy()
        self.isCali = [isScale, isBias_constant, isBias_trend, isBias_quadratic]
        #======= value ======
        scale = self.accConfig.cali_value['Scale'].copy()
        bias_constant = self.accConfig.cali_value['Bias_constant'].copy()
        bias_trend = self.accConfig.cali_value['Bias_trend'].copy()
        bias_quadratic = self.accConfig.cali_value['Bias_quadratic'].copy()
        self.scale = np.array(scale, dtype=np.float64).reshape((3, 3))
        self.bias = np.array([bias_constant, bias_trend, bias_quadratic], dtype=np.float64)

        cali = []
        for term in self.isCali:
            cali = cali + term
        par = list(self.scale.flatten()) + list(self.bias.flatten())
        par = np.array(par)[cali]
        return par.copy()

    def getPar(self):
        return self.__par


class Accelerometer:
    """
    Acquire the acceleration and dadp for the given time period.The current implementation of the accelerometer assumes
    the scale/bias is estimated per arc, however, this is inflexible and must be fixed in the future.

    TODO: use a more advanced way to analyze accelerometer (design matrix).
    TODO: And smaller arc (like bias @0.5 hour) should be introduced to make solution more flexible.
    """

    def __init__(self, rd: ArcData, ap: AccCaliPar, **kwargs):
        """
        acc and sca are the array with their first column recording the same timestamp [gps second]
        :param sat: define the sat ID for the accelerometer to be dealt with.
        :param acc: original data set of accelerometers [m/s]
        :param sca: original data set of star camera quaternions.
        :param kwargs: define which parameters to be calibrated. [3-axis]
        """
        '''assure the consistency of acc and sca data'''
        self.__rd = rd
        self.__sat = ap.sat
        self.__arcNo = ap.arcNo
        self.__bias = ap.bias
        self.__scale = ap.scale
        self.__isCali = ap.isCali
        self.__accConfig = ap.accConfig
        self.__acc = self.__rd.getData(arc=self.__arcNo, kind=Payload.ACC, sat=self.__sat)
        self.__sca = self.__rd.getData(arc=self.__arcNo, kind=Payload.SCA, sat=self.__sat)
        # commonTime = set(self.__acc[:, 0].astype(np.int64))
        # commonTime = commonTime & set(self.__sca[:, 0].astype(np.int64))
        # commonTime = np.array(list(commonTime))
        # commonTime.sort()
        # index1 = [list(self.__acc[:, 0]).index(x) for x in commonTime]
        # index2 = [list(self.__sca[:, 0]).index(x) for x in commonTime]
        # self.__acc = self.__acc[index1, :]
        # self.__sca = self.__sca[index2, :]
        assert np.max(np.abs(self.__acc[:, 0] - self.__sca[:, 0])) < Pre_Constants.num_tolerance
        self.__dataTemp = kwargs.get('dataTemp', ap.accConfig.temp_non_conservative_data)
        os.makedirs(self.__dataTemp, exist_ok=True)
        '''Above defines a fully populated scale matrix (span to 1-d)'''
        '''
        for example
        [true, false, false,
         false, true, false,
         false, false, true]
        '''
        '''
        such a fully populated matrix considers the correlation of the 3-axis acc measurements
        '''
        index = [2, 3, 4, 1]
        q = Quaternion.normalize(self.__sca[:, index])
        self.__sca = Quat(q=q).transform
        pass

    # @DeprecationWarning
    # def setPar(self, **kwargs):
    #     """
    #     set parameters from json file
    #     :param kwargs: define the value of the parameters. [3-axis]
    #     :return:
    #     """
    #     scale = kwargs.get('Scale', Pre_Accelerometer.cali_value['Scale'])
    #     # scale = kwargs.get('Scale', [0.9465, 0.9842, 0.9303])  # for sat B
    #     bias_constant = kwargs.get('Bias_constant', Pre_Accelerometer.cali_value['Bias_constant'])
    #     bias_trend = kwargs.get('Bias_trend', Pre_Accelerometer.cali_value['Bias_trend'])
    #     bias_quadratic = kwargs.get('Bias_quadratic', Pre_Accelerometer.cali_value['Bias_quadratic'])
    #
    #     self.__scale = np.array(scale, dtype=np.float).reshape((3, 3))
    #     self.__bias = np.array([bias_constant, bias_trend, bias_quadratic], dtype=np.float)
    #     return self

    # @DeprecationWarning
    def setParFromList(self, paralist:list):
        """
        set parameters from a list; update the parameters: scale & bias
        :param paralist: 0:scale,1:bias. bias: [1-d], scale:[1-d]
        :return:
        """
        scale = paralist[0]
        bias = paralist[1]
        self.__scale = scale.reshape((3, 3))
        self.__bias = bias.reshape((3, 3))
        return self

    def updatePar(self, paralist):
        """
        set parameters from a list of calibrated parameters; update the parameters: scale & bias
        :param paralist: define the value of the parameters. [1-d]
        :return:
        """
        cali = []
        for term in self.__isCali:
            cali = cali + term

        valuelist = np.zeros(18, dtype=np.float64)
        valuelist[cali] = np.array(paralist)

        scale = self.__scale.flatten()
        bias = self.__bias.flatten()
        scale[cali[0:9]] = valuelist[0:9][cali[0:9]]
        bias[cali[9:]] = valuelist[9:][cali[9:]]

        self.__scale = scale.reshape((3, 3))
        self.__bias = bias.reshape((3, 3))
        return self

    def get_acceleration(self):
        """
        Non-conservative force measured by accelerometers and star camera. [ICRS][m/s]
        acc = q*(b+E*a)  : q--sca , b--bias, E--scale
        :return: result for all the time list
        """
        acc_old = self.__acc[:, 1:]
        time = self.__acc[:, 0]
        dt = (time - time[0]) / 60
        '''x-axi'''
        bias_x = self.__bias[0][0] + self.__bias[1][0] * dt + self.__bias[2][0] * (dt ** 2)
        '''y-axi'''
        bias_y = self.__bias[0][1] + self.__bias[1][1] * dt + self.__bias[2][1] * (dt ** 2)
        '''z-axi'''
        bias_z = self.__bias[0][2] + self.__bias[1][2] * dt + self.__bias[2][2] * (dt ** 2)
        bias = np.array([bias_x, bias_y, bias_z]).transpose()

        acc = self.__scale @ acc_old[:, :, None] + bias[:, :, None]
        acc = self.__sca @ acc

        # acc = self.__sca @ acc_old[:, :, None]
        return acc[:, :, 0]

    def get_acceleration_test(self):
        return self.__acc[:, 1:]

    def get_du2p(self):
        """
        Derivative of acc with respect to the parameters such as bias & scale.
        :return: result for all the time list
        """
        time = self.__acc[:, 0]
        dt = (time - time[0]) / 60
        q = self.__sca

        acc_old = self.__acc[:, 1:]
        acc = acc_old[:, None, :]
        '''scale factor'''
        ax = q[:, 0, :][:, :, None] @ acc
        ay = q[:, 1, :][:, :, None] @ acc
        az = q[:, 2, :][:, :, None] @ acc
        a1 = np.zeros(shape=(len(time), 3, 9), dtype=float)
        a1[:, 0, :] = ax.reshape((-1, 9))
        a1[:, 1, :] = ay.reshape((-1, 9))
        a1[:, 2, :] = az.reshape((-1, 9))
        '''bias constant'''
        a2 = q
        '''bias trend'''
        a3 = q * dt[:, None, None]
        '''bias quadratic'''
        a4 = q * dt[:, None, None] ** 2

        isScale, isBias_constant, isBias_trend, isBias_quadratic = self.__isCali
        b1 = a1[:, :, isScale]
        b2 = a2[:, :, isBias_constant]
        b3 = a3[:, :, isBias_trend]
        b4 = a4[:, :, isBias_quadratic]
        dadp = np.zeros((b1.shape[0], b1.shape[1], b1.shape[2] + b2.shape[2] + b3.shape[2] + b4.shape[2]))

        if dadp.shape[2] == 0:
            return None

        '''sort by the order --- [scale, bias_constant, bias_trend, bias_quadratic]'''
        m = 0
        n = b1.shape[2]
        dadp[:, :, m:n] = b1
        m = n
        n += b2.shape[2]
        dadp[:, :, m:n] = b2
        m = n
        n += b3.shape[2]
        dadp[:, :, m:n] = b3
        m = n
        n += b4.shape[2]
        dadp[:, :, m:n] = b4

        return dadp

    # def get_timestamp(self):
    #     """
    #
    #     :return: gps [second] time stamp for the time list
    #     """
    #     return self.__acc[:, 0]

    def get_and_save(self):
        sat = self.__sat
        time = self.__acc[:, 0]
        date_span = self.__rd.date_span

        filename = pathlib.Path(self.__dataTemp).joinpath(
            date_span[0] + '_' + date_span[1] + '_' + sat + '_' + str(self.__arcNo) + '.hdf5')
        wr = h5py.File(filename, "w")

        acc = self.get_acceleration()
        dadp = self.get_du2p()

        wr.create_dataset('time', data=time)
        wr.create_dataset('acc', data=acc)
        wr.create_dataset('bias', data=self.__bias)
        wr.create_dataset('dadp', data=dadp)
        wr.close()
        pass


