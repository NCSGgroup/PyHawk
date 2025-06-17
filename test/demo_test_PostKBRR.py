# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2025/5/21 17:01
# @File    : demo_test_PostKBRR.py
# @Software: PyCharm

import h5py
import numpy as np
import matplotlib.pyplot as plt

def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])

filepath1 = '../temp/PostKBRR/2009-05-01-2009-05-31/'
filepath2 = '../temp/PostOrbit/2009-05-01-2009-05-31/'

orbit = []
kbr = []
post_rms = []
for i in range(0, 247):
    filename1 = filepath1 + 'RangeRate_' + str(i) + '.hdf5'
    h51 = h5py.File(filename1, 'r')
    post_kbrr = h51['post_kbrr'][:]
    kbrr = rms(post_kbrr)

    filename2 = filepath2 + 'Orbit_' + str(i) + 'A.hdf5'
    h5 = h5py.File(filename2, 'r')
    # leo = rms(h5['post_Orbit'][:]) * 20
    leo = 0.02
    kbr.append(kbrr)
    orbit.append(leo)
    factor = (leo / kbrr) ** 2
    post_rms.append(factor)

print(post_rms)
# plt.figure(dpi=300, figsize=(16, 8))
plt.title('KBRR-RMS-{}-{}'.format('2009-05-01', '2009-05-31'), fontsize=24)
plt.plot(np.arange(len(post_rms)), [i * 1000000 for i in post_rms], marker='o', label="arc post RMS")
plt.yticks(fontsize=18)
plt.ylim(0.15, 0.5)
plt.ylabel(r'$X/um$', fontsize=20)
plt.legend(loc='upper right', fontsize=20)
plt.show()
# plt.savefig('../result/img/post-KBRR-{}-{}.png'.format('2009-05-01', '2009-05-31'))