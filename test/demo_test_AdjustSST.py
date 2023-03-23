from src.Solver.AdjustSST import RangeRate
import h5py
import pathlib
import matplotlib.pyplot as plt
import src.Preference.Pre_AdjustOrbit as Pre_AdjustOrbit
import numpy as np


def demo1():
    RR = RangeRate(0)
    RR.configure()
    RR.calibrate()
    pass


def demo2():
    date_begin = '2008-01-01'
    date_end = '2008-01-31'
    rms_list = []
    for i in range(0, 108):
        RR = RangeRate(i)
        RR.configure()
        RR.calibrate()
        res_dir = pathlib.Path(Pre_AdjustOrbit.PathOfFiles.RangeRateTemp).joinpath(date_begin + '_' + date_end)
        res_filename = res_dir.joinpath('RangeRate_' + str(i) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        postfit = h5['post_residual'][:]
        rms_list.append(rms(postfit))
    np.savez('RangeRateRMS.npz', RMS=rms_list)
    pass


def draw1():
    date_begin = '2008-01-01'
    date_end = '2008-01-31'
    arcNo = 0

    plt.style.use(['grid'])
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
    ax.plot(t1, prefit, label='prefit')

    ax.legend()
    plt.show()
    pass


def drawRMS():
    R = np.load("RangeRateRMS.npz")
    RMS = R['RMS']

    plt.style.use(['grid'])
    plt.title('RangeRateRMS(2008-1-1~2008-1-31)', fontsize=24)
    plt.plot(np.arange(len(RMS)), RMS, label="PostResidualsRMS")
    plt.xlabel('arcNo', fontsize=20)
    plt.ylabel(r'$m$', fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize=20)
    # plt.savefig('../result/img/{}-{}.png'.format(length[0], length[-1]), dpi=300)
    # plt.grid(ls='--')
    plt.show()
    pass


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


if __name__ == '__main__':
    # demo2()
    draw1()
    # drawRMS()