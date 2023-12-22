import numpy as np
import os
import socket
import h5py



res_filename = '1.hdf5'
Data = h5py.File(res_filename, 'r')
print(Data.keys())
Data.close()

# oustname = socket.gethostname()
# print('1',oustname)
#
# arr = np.zeros(10)
# print(arr)