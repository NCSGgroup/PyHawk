from src.ForceModel.BaseGravCS import BaseGravCS
from src.Interface.LoadSH import LoadSH
from src.Preference.Pre_ForceModel import RefGravModelConfig


class AbsRefGravModel(BaseGravCS):

    def __init__(self):
        super(AbsRefGravModel, self).__init__()


class RefGravModel(AbsRefGravModel):

    def __init__(self, type=RefGravModelConfig.type):
        super(RefGravModel, self).__init__()
        self.__model = None
        self.__Nmax = None
        self.__type = type

    def setModel(self, model: LoadSH, Nmax=RefGravModelConfig.Nmax):
        """
        :param Nmax: max degree to be returned
        :param model:
        :return:
        """
        self.__model = model
        self.__Nmax = Nmax
        return self

    def getCS(self):
        """
        :param t: time of reference model to be returned (Considering that the reference model might be
        changing with time), [GPS time in format of MJD]
        :return:
        """
        epoch = self._time.getGPS_from_epoch2000
        if self.__type == 0:
            '''constant type， a constant reference model without changing with time.'''
            return self.getModelPar().getCS(self.__Nmax)

    def getModelPar(self) -> LoadSH:
        """

        :return: the model associated with the GM, radius or the other constants
        """
        return self.__model

