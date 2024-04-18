import numpy as np
import os
import socket
import h5py
from scipy import linalg


a = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]])
w = linalg.cholesky(a)

print(w.T.dot(w))