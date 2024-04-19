import h5py
import pathlib


res_dir = pathlib.Path('../temp/OrbitAdjustment/2021-01-01_2021-01-31')
res_filename = res_dir.joinpath(str(43) + '_AB.hdf5')
h5 = h5py.File(res_filename, 'r')
acc =h5['Accelerometer'][:]
print(res_filename)
print(h5.keys())
