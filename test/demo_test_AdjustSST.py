import sys
sys.path.append('../')
from src.Solver.AdjustSST import RangeRate
import h5py
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.fft import fft
from numpy.linalg import norm


def demo1():
    RR = RangeRate(0)
    RR.configure()
    RR.calibrate()
    pass


def demo2():

    for i in range(38, 243):
        print(i)
        RR = RangeRate(i)
        RR.configure()
        RR.calibrate()
    pass


def draw1():
    date_begin = '2008-01-01'
    date_end = '2008-01-31'
    arcNo = 0

    # plt.style.use(['grid'])
    # plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='GPS time [sec]', ylabel=r'$m$')

    res_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.RangeRateTemp).joinpath(date_begin + '_' + date_end)
    res_filename = res_dir.joinpath('RangeRate_' + str(arcNo) + '.hdf5')
    h5 = h5py.File(res_filename, 'r')
    t1 = h5['pre_t'][()]
    t2 = h5['post_t'][()]
    prefit = h5['pre_residual'][()]
    postfit = h5['post_residual'][()]

    ax.plot(t2, postfit, label='postfit')
    # ax.plot(t1, prefit, label='prefit')

    ax.legend()
    plt.show()
    pass


def drawRMS():
    data = np.load("2008-01.npz")
    pre_rms = data['pre']
    post_rms = data['post']
    # RMS = R['RMS']
    # RMS01 = np.load('2008-01.npy')[:239]
    # RMS02 = np.load('2008-01-arc6-one.npy')
    # diff = RMS02 - RMS01
    # date_begin = '2008-01-01'
    # date_end = '2008-01-31'
    # res_dir = pathlib.Path('../temp/RangeRate').joinpath(date_begin + '_' + date_end)
    # pre_RMS = []
    # post_RMS = []
    # for i in range(0, 121):
    #     res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
    #     if os.path.exists(res_filename):
    #         h5 = h5py.File(res_filename, 'r')
    #         print(h5.keys())
    #         if len(h5.keys()) != 0:
    #             prefit = h5['pre_residual'][()]
    #             postfit = h5['post_residual'][()]
    #             pre_RMS.append(rms(prefit))
    #             post_RMS.append(rms(postfit))

    # RMS01.extend(RMS02)

    # np.savez('2008-01.npz', pre=pre_RMS, post=post_RMS)

    plt.style.use(['grid'])
    plt.title('RangeRateRMS(2008-1-1~2008-1-31)', fontsize=24)

    # plt.plot(np.arange(len(pre_rms)), pre_rms, marker='o',label="pre_RMS")
    plt.plot(np.arange(len(post_rms)), post_rms, marker='o', label="post_RMS")
    plt.xlabel('Arc number', fontsize=20)
    plt.ylabel(r'Residual($m/s$)', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    plt.grid(ls='--')
    plt.show()
    pass


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


def psd():
    date_begin = '2008-01-01'
    date_end = '2008-01-31'
    res_dir = pathlib.Path('../temp/RangeRate').joinpath(date_begin + '_' + date_end)
    pre_data = []
    post_data = []
    for i in range(0, 121):
        res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
        if os.path.exists(res_filename):
            h5 = h5py.File(res_filename, 'r')
            print(h5.keys())
            if len(h5.keys()) != 0:
                prefit = h5['pre_residual'][()]
                postfit = h5['post_residual'][()]
                pre_data = pre_data + list(prefit)
                post_data = post_data + list(postfit)

    N = len(pre_data)
    Pxx = []
    dt = 5
    fs = 1 / dt
    nfft = N
    for i in range(N//nfft):
        w = np.hanning(nfft)
        p = np.abs(fft(w * pre_data[i*nfft:(i+1)*nfft])) ** 2 / fs / norm(w)**2
        Pxx.append(p)
    Pxx2 = np.mean(Pxx, axis=0)

    f2 = np.array([k*fs/nfft for k in range(nfft//2)])

    np.savez('2008-01-density-pre.npz', f2=f2, Pxx2=Pxx2)
    pass


def drawPSD():
    data = np.load("2008-01-density-pre.npz")
    data2 = np.load("2008-01-density-post.npz")
    f2 = data2['f2']
    Pxx2 = data2['Pxx2']

    f1 = data['f2']
    Pxx1 = data['Pxx2']
    nfft1 = len(Pxx1)
    nfft2 = len(Pxx2)
    plt.style.use(['grid'])
    plt.title('RangeRateRMS(2008-1-1~2008-1-31)', fontsize=24)

    # plt.plot(np.arange(len(pre_rms)), pre_rms, marker='o',label="pre_RMS")
    plt.plot(f1, Pxx1[nfft1 // 2 +1:], 'b--', label="pre_RMS", color='green')
    plt.plot(f2, Pxx2[nfft2 // 2:], 'b--', label="post_RMS")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('f/Hz', fontsize=20)
    plt.ylabel('PSD', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    plt.tick_params(top=True, right=True, direction='in', which='both')
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    plt.grid(ls='--')
    plt.show()
    pass


if __name__ == '__main__':
    # demo2()
    # drawRMS()
    # draw1()
    # psd()
    drawPSD()