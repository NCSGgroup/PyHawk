import h5py
import pathlib
import numpy as np
import src.Preference.Pre_AdjustOrbit as Pre_AdjustOrbit


class AdjustOrbit:
    def __init__(self):
        self._state_dir = None
        self.configure()

    def configure(self, **kwargs):
        self._state_dir = kwargs.get('state_dir', Pre_AdjustOrbit.PathOfFiles.StateVectorDataTemp)

        """Define the directory to deposit the state vectors"""
        self._state_dir = self._state_dir.joinpath(date_begin + '_' + date_end)
        if not self._state_dir.is_dir():
            self._state_dir.mkdir(exist_ok=True)

    def _save_StateVectors(self, state):
        res_dir = self._state_dir
        res_filename = res_dir.joinpath(str(self._arcNo) + '_' + self._sat.name + '.hdf5')
        h5 = h5py.File(res_filename, 'w')
        h5.create_dataset('t', data=state[0])
        h5.create_dataset('r', data=state[1])
        h5.create_dataset('v', data=state[2])
        h5.close()
        pass