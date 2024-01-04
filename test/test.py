import numpy as np
import os
import socket
import h5py


orbit_name = 'Orbit_0_AB.hdf5'
adjust_name = '0_AB.hdf5'


orbit = h5py.File(orbit_name, 'r')

orbit_AB = orbit['r'][:]
orbit_A = orbit_AB[:, 0:3]
orbit_B = orbit_AB[:, 3:6]

adjust = h5py.File(adjust_name, 'r')
adjust_AB = adjust['r'][:]
adjust_A = adjust_AB[:, 0:3, 0]
adjust_B = adjust_AB[:, 3:6, 0]

diff = orbit_A - adjust_A
print(diff)


