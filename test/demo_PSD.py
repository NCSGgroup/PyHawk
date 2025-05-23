# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2025/4/9 15:05
# @File    : demo_PSD.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
res_6h = h5py.File("../temp/RangeRate/acc-6h/6par/3sigma/2009-05-01_2009-05-31.hdf5", "r")
signal_with_noise_6h = res_6h['postfit'][:] * 1e7
fs = 0.2
npe = int(fs * len(signal_with_noise_6h))
frequencies_welch_6h, psd_welch_6h = signal.welch(signal_with_noise_6h, fs, nperseg=npe)

res = h5py.File("../temp/RangeRate/acc-2h/kin+gnv/2009-05-01_2009-05-31.hdf5", "r")
signal_with_noise = res['postfit'][:] * 1e7

frequencies_welch, psd_welch = signal.welch(signal_with_noise, fs, nperseg=npe)

res_3h = h5py.File("../temp/RangeRate/acc-1.5h/2009-05-01_2009-05-31.hdf5", "r")
signal_with_noise_3h = res_3h['postfit'][:] * 1e7

frequencies_welch_3h, psd_welch_3h = signal.welch(signal_with_noise_3h, fs, nperseg=npe)

plt.plot(frequencies_welch_6h, psd_welch_6h, lw=1.5, alpha=0.8, color='black', label='6h')
plt.plot(frequencies_welch, psd_welch, lw=1.5, alpha=0.8, color='red', label='2h')
plt.plot(frequencies_welch_3h, psd_welch_3h, lw=1.5, alpha=0.8, color='green', label='1.5h')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, linestyle='dashed')
font = {'size': 20}
plt.xlabel('Frequency [Hz]', fontdict=font)
plt.ylabel(r'$\sqrt{PSD}$ ($arcsec /\sqrt{Hz}$)', fontdict=font)
# plt.tick_params(direction='in', top=True, bottom=True, left=True)
# plt.tick_params(axis='y', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
# plt.tick_params(axis='x', direction='in', top=True, bottom=True, left=True, right=True, width=1.5, labelsize=20)
plt.legend(loc='upper right', fontsize='24')
# 调整子图布局
# plt.tight_layout()
plt.show()