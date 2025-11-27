# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2025/4/9 15:05
# @File    : demo_PSD.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
res_6h = h5py.File("../result/xyz/fle/test5/pre-2019-01-01_2019-01-31.hdf5", "r")
signal_with_noise_6h = res_6h['prefit'][:] * 1e9
fs = 0.2
npe = int(fs * len(signal_with_noise_6h))
frequencies_welch_6h, psd_welch_6h = signal.welch(signal_with_noise_6h, fs, nperseg=npe)

res = h5py.File("../result/IGG-2019-01-01_2019-01-31.hdf5", "r")
signal_with_noise = res['prefit'][:] * 1e9
frequencies_welch, psd_welch = signal.welch(signal_with_noise, fs, nperseg=npe)

res_3h = h5py.File("../result/butter/0.03/post-2019-05-01_2019-05-31.hdf5", "r")
signal_with_noise_3h = res_3h['postfit'][:] * 1e9
frequencies_welch_3h, psd_welch_3h = signal.welch(signal_with_noise_3h, fs, nperseg=npe)

res_4h = h5py.File("../result/butter/0.05/post-2019-05-01_2019-05-31.hdf5", "r")
signal_with_noise_4h = res_4h['postfit'][:] * 1e9

frequencies_welch_4h, psd_welch_4h = signal.welch(signal_with_noise_4h, fs, nperseg=npe)

plt.plot(frequencies_welch_6h, psd_welch_6h, lw=1.5, alpha=0.8, color='black', label='pyhawk')
plt.plot(frequencies_welch, psd_welch, lw=1.5, alpha=0.8, color='red', label='IGG')
# plt.plot(frequencies_welch_3h, psd_welch_3h, lw=1.5, alpha=0.8, color='green', label='0.03')
# plt.plot(frequencies_welch_4h, psd_welch_4h, lw=1.5, alpha=0.8, color='blue', label='0.05')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, linestyle='dashed')
font = {'size': 20}
plt.xlabel('Frequency [Hz]', fontdict=font)
plt.ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
# plt.tick_params(direction='in', top=True, bottom=True, left=True)
# plt.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
# plt.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
plt.legend(loc=2, fontsize='20')
# 调整子图布局
# plt.tight_layout()
plt.show()