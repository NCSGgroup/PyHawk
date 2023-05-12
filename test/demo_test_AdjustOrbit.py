import src.Preference.backups.Pre_AdjustOrbit as Pre_AdjustOrbit
import pathlib
import h5py


def demo1():
    date_span = ['2008-01-01', '2008-01-31']
    res_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.OrbitAdjustResTemp).joinpath(date_span[0] + '_' + date_span[1])
    res_filename = res_dir.joinpath(str(0) + '_AB.hdf5')
    h5 = h5py.File(res_filename, 'r')
    print(h5.keys())
    print(h5['TransitionMatrix'][:])
    print(h5['Accelerometer'][:])


if __name__ == '__main__':
    demo1()