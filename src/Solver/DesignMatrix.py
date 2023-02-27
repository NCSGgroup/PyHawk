import src.Preference.Pre_Interface as Pre_Interface
import src.Preference.Pre_Solver as Pre_Solver
from src.Preference.EnumType import SSTObserve
from src.Solver.AdjustOrbit import AdjustOrbitForDM
from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff


class GravityDesignMat:
    """
    Gravity inversion in terms of spherical harmonic.
    """
    def __init__(self, arcNo: int, date_span=Pre_Interface.date_span):
        self._arcNo = arcNo
        self._date_span = date_span
        pass

    def configure(self):
        pass

    def orbit_integration(self):
        adjustDM = AdjustOrbitForDM()
        adjustDM.configure(Orbit_var_2nd_diff)
        '''define the directory of res output'''
        res_dir = Pre_Solver.PathOfFiles.DesignMatrixTemp
        res_dir = res_dir.joinpath(self._date_span[0] + '_' + self._date_span[1])
        if not res_dir.is_dir():
            res_dir.mkdir(exist_ok=True)
        kwargs = {'ChangeTempPath': res_dir}
        adjustDM.updateInit().calibrate(**kwargs)

        return self

    def get_orbit_design_matrix(self):
        pass

    def get_sst_design_matrix(self):
        """
        Obtain the SST design matrix with the orbit design matrix.
        :return:
        """
        SSTOption = Pre_Solver.GravityInversion.SSTOption

        if SSTOption == SSTObserve.Range:
            self.__getRangeDM()
        elif SSTOption == SSTObserve.RangeRate:
            self.__getRangeRateDM()
        elif SSTOption == SSTObserve.RangeAcc:
            self.__getRangeAccDM()

    def __getRangeDM(self):
        pass

    def __getRangeRateDM(self):
        pass

    def __getRangeAccDM(self):
        pass

