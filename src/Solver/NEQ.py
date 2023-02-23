import src.Preference.Pre_Interface as Pre_Interface
import src.Preference.Pre_NEQ as Pre_NEQ
import pathlib
import h5py


class GravityNEQperArc:
    """
    Gravity inversion in terms of spherical harmonic.
    Retrieve the design matrix and observations, then transform the observation-equation into the normal equations.
    """
    def __init__(self, arcNo: int, date_span=Pre_Interface.date_span):
        self.__date_begin = date_span[0]
        self.__date_end = date_span[1]
        self.__arcNo = arcNo

        res_dir = pathlib.Path(Pre_NEQ.PathOfFiles.NormalEqTemp).joinpath(date_span[0] + '_' + date_span[1])
        if not res_dir.is_dir():
            res_dir.mkdir(exist_ok=True)
        res_filename = res_dir.joinpath(str(arcNo) + '.hdf5')
        self.__saveData = h5py.File(res_filename, 'w')
        pass

    def configure(self, **kwargs):
        pass

    def get_neq(self):
        """
        three steps to acquire the normal equations. Assure the 'configure' is called before.
        :return:
        """
        pass

    def get_neq_orbit(self):
        """
        get Normal Equations for the orbit only
        :return:
        """
        pass





