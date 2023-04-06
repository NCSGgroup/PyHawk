from src.ForceModel.BaseGravCS import BaseGravCS
from src.Interface.LoadSH import LoadSH, LoadGif48
from src.Preference.Pre_ForceModel import RefGravModelConfig, PathOfFiles
from src.Preference.EnumType import StaticGravModel
from src.Preference.ForceModelConfig import ForceModelConfig


class AbsRefGravModel(BaseGravCS):

    def __init__(self):
        super(AbsRefGravModel, self).__init__()


class RefGravModel(AbsRefGravModel):

    def __init__(self):
        super(RefGravModel, self).__init__()
        self.__model = None
        self.__Nmax = None
        self.__kind = None
        self._FMConfig = None
        self.__staticModel = None

    def configure(self, FMConfig: ForceModelConfig):
        self._FMConfig = FMConfig
        '''config refGravModel'''
        self._RefGravModelConfig = self._FMConfig.RefGravModel()
        self._RefGravModelConfig.__dict__ = self._FMConfig.RefGravModelConfig
        self.__kind = self._RefGravModelConfig.kind
        staticModel = self._RefGravModelConfig.StaticModel
        '''config path'''
        self._path = self._FMConfig.PathOfFiles()
        self._path.__dict__ = self._FMConfig.PathOfFilesConfig
        if staticModel == StaticGravModel.Gif48.name:
            self.__staticModel = LoadGif48().load(self._path.Gif48)
        elif staticModel == StaticGravModel.EIGEN6_C4.name:
            self.__staticModel = LoadGif48().load(self._path.EIGEN6_C4)
        elif staticModel == StaticGravModel.GOCO02s.name:
            self.__staticModel = LoadGif48().load(self._path.GOCO02s)
        return self

    def getStaticModel(self):
        return self.__staticModel

    def setModel(self, model: LoadSH):
        """
        :param Nmax: max degree to be returned
        :param model:
        :return:
        """
        self.__model = model
        self.__Nmax = self._RefGravModelConfig.Nmax
        return self

    def getCS(self):
        """
        :param t: time of reference model to be returned (Considering that the reference model might be
        changing with time), [GPS time in format of MJD]
        :return:
        """
        epoch = self._time.getGPS_from_epoch2000
        if self.__kind == 0:
            '''constant type， a constant reference model without changing with time.'''
            return self.getModelPar().getCS(self.__Nmax)

    def getModelPar(self) -> LoadSH:
        """

        :return: the model associated with the GM, radius or the other constants
        """
        return self.__model

