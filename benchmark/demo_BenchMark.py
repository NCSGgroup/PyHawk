# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:54
# @File    : demo_BenchMark.py
# @Software: PyCharm
from benchmark.BenchMarkForJson import BenchMark
from src.Preference.Pre_ForceModel import ForceModelConfig
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import json
import matplotlib


def demo1():
    FMConfig = ForceModelConfig()
    benchMark = BenchMark()
    '''three body'''
    threeBody = json.load(open('../setting/BenchMarkConfig/BenchMark_ThreeBody.json', 'r'))
    FMConfig.__dict__ = threeBody
    benchMark.three_body(FMConfig)
    '''AOD'''
    aod = json.load(open('../setting/BenchMarkConfig/BenchMark_AOD.json', 'r'))
    FMConfig.__dict__ = aod
    benchMark.aod(FMConfig)
    '''relativistic'''
    relativistic = json.load(open('../setting/BenchMarkConfig/BenchMark_Relativity.json', 'r'))
    FMConfig.__dict__ = relativistic
    benchMark.relativistic(FMConfig)
    '''eot11a'''
    eot11a = json.load(open('../setting/BenchMarkConfig/BenchMark_EOT11a.json', 'r'))
    FMConfig.__dict__ = eot11a
    benchMark.eot11a(FMConfig)
    '''FES2014'''
    fes2014 = json.load(open('../setting/BenchMarkConfig/BenchMark_FES2104.json', 'r'))
    FMConfig.__dict__ = fes2014
    benchMark.fes2014b(FMConfig)
    '''Atmos tide'''
    atmos = json.load(open('../setting/BenchMarkConfig/BenchMark_Atmos.json', 'r'))
    FMConfig.__dict__ = atmos
    benchMark.atmos_tide(FMConfig)
    '''gravity field'''
    gravity_field = json.load(open('../setting/BenchMarkConfig/BenchMark_Gravity.json', 'r'))
    FMConfig.__dict__ = gravity_field
    benchMark.gravity_field(FMConfig)
    '''solidEarth tide'''
    solid_earth = json.load(open('../setting/BenchMarkConfig/BenchMark_SolidEarth.json', 'r'))
    FMConfig.__dict__ = solid_earth
    benchMark.solid_earth(FMConfig)
    '''solidEarth pole tide'''
    earth_pole = json.load(open('../setting/BenchMarkConfig/BenchMark_EarthPole.json', 'r'))
    FMConfig.__dict__ = earth_pole
    benchMark.earth_pole(FMConfig)
    '''ocean pole tide'''
    ocean_pole = json.load(open('../setting/BenchMarkConfig/BenchMark_OceanPole.json', 'r'))
    FMConfig.__dict__ = ocean_pole
    benchMark.ocean_pole(FMConfig)
    pass


def demo2():
    diff = np.load('../result/benchmarktest/sun.npy')
    timeList = np.shape(diff)[0]
    diff = np.array(diff)
    ax = diff[:, 0]
    ay = diff[:, 1]
    az = diff[:, 2]
    plt.style.use('grid')
    rcParams['font.size'] = 24  # 设置字体大小为14
    rcParams['font.family'] = 'serif'  # 设置字体为衬线字体

    plt.title('Sun', fontsize=24)
    plt.plot(ax, color='#dd2c00', label='x')
    plt.plot(ay, color='#00695c', label='y')
    plt.plot(az, color='#1a237e', label='z')
    plt.legend()
    plt.xlabel('MJD', fontsize=24)
    plt.ylabel(r'$m/s^2$', fontsize=24)
    # plt.savefig('../result/benchmarktest/AODRL06.png')
    plt.show()


def plot():
    matplotlib.rcParams.update({'font.size': 14})
    savedir = '../result/benchmarktest/'
    res = {
        'Moon': np.load(savedir + 'moon.npy'),
        'Sun': np.load(savedir + 'sun.npy'),
        'Planets': np.load(savedir + 'planets.npy'),
        'relativistic': np.load(savedir + 'relativistic.npy'),
        'EOT11a': np.load(savedir + 'eot11a.npy'),
        'FES2014b': np.load(savedir + 'fes2014.npy'),
        'Atmos tide': np.load(savedir + 'atmos.npy'),
        'dealiasing': np.load(savedir + 'AODRL06.npy'),
        'pole tide': np.load(savedir + 'earthpole.npy'),
        'ocean pole': np.load(savedir + 'oceanpole.npy'),
        'solid tide': np.load(savedir + 'solidearth.npy'),
        'Earth gravity': np.load(savedir + 'gravity.npy')
    }
    epoch = np.linspace(54650, 54651, 2880)
    plt.style.use(['grid'])

    ns = {}
    ds = {}
    for key in res.keys():
        ns[key] = np.linalg.norm(res[key], axis=1)
        ds[key] = ns[key].max()

    colors = {
        'Moon': 'cyan',
        'Sun': 'lightblue',
        'Planets': 'darkblue',
        'relativistic': 'gold',

        'FES2014b': 'grey',
        'Atmos tide': 'forestgreen',
        'dealiasing': 'red',

        'pole tide': 'lightgrey',
        'ocean pole': 'yellow',
        'solid tide': 'brown',
        'Earth gravity': 'black'
    }
    plt.figure()
    # plt.axhline(y=1e-11, color='k', linestyle='--', label=r'$10^{-11} m/s^2$', linewidth=4)

    # for key in ns.keys():
    #     if key == 'EOT11a': continue
    #     plt.plot(epoch, ns[key], label=key, color=colors[key])
    #
    # plt.xlim([epoch[0], epoch[-1]])
    # plt.ylim([1e-26, 1e-10])
    # plt.xlabel(xlabel='MJD')
    # plt.ylabel(ylabel=r'$m/s^2$')
    # plt.yscale('log')
    # plt.title('Benchmark test with PyHawk')
    # plt.yticks([1e-10, 1e-12, 1e-14, 1e-16, 1e-18,1e-20, 1e-22,1e-24,1e-26])
    # leg = plt.legend(bbox_to_anchor=(1, 0.8), loc=2, borderaxespad=0,numpoints=1,fontsize=10)
    # for line in leg.get_lines():
    #     line.set_linewidth(4.0)

    ds.pop('EOT11a')
    namelist = ['Earth gravity', 'Moon', 'Sun', 'Planets',
                'solid tide', 'FES2014b', 'relativistic', 'dealiasing', 'pole tide', 'Atmos tide', 'ocean pole']
    vv = [ds[k] for k in namelist]
    bernese = [2e-15, 3e-16, 0.8e-16, 0.8e-17, 2e-13, 1.5e-13, 1e-20, 3e-14, 4e-15, 6e-19, 0.9e-15]

    plt.scatter(np.arange(11), vv, marker='*', s=64, c='r', label='PyHawk')
    plt.scatter(np.arange(11), bernese, marker='v', c='darkblue', label='Bernese', s=64)

    plt.ylim([1e-24, 1e-10])
    plt.ylabel(ylabel=r'$m/s^2$')
    plt.xlabel(xlabel='MJD')
    plt.yscale('log')
    plt.title('Benchmark test with PyHawk')
    plt.axhline(y=1e-11, color='k', linestyle='--', label=r'$10^{-11} m/s^2$', linewidth=4)
    plt.legend(bbox_to_anchor=(1, 0.8), loc=2, frameon=False)
    plt.xticks(np.arange(11),namelist,rotation=45)
    plt.show()
    pass

def plot_benchmark1():
    savedir = '../result/benchmarktest/'
    res = {
        'Moon': np.load(savedir + 'moon.npy'),
        'Sun': np.load(savedir + 'sun.npy'),
        'Planets': np.load(savedir + 'planets.npy'),
        'relativistic': np.load(savedir + 'relativistic.npy'),

        'EOT11a': np.load(savedir + 'eot11a.npy'),
        'FES2014b': np.load(savedir + 'fes2014.npy'),
        'Atmos tide': np.load(savedir + 'atmos.npy'),
        'dealiasing': np.load(savedir + 'AODRL06.npy'),

        'pole tide': np.load(savedir + 'earthpole.npy'),
        'ocean pole': np.load(savedir + 'oceanpole.npy'),
        'solid tide': np.load(savedir + 'solidearth.npy'),
        'Earth gravity': np.load(savedir + 'gravity.npy')
    }
    savedir = '../result/benchmarktest/'
    epoch = np.linspace(54650, 54651, 2880)
    # plt.style.use(['ieee', 'grid'])
    fig = plt.figure()

    keylist = list(res.keys())
    for i in range(8, 12):
        key = keylist[i]
        ax = fig.add_subplot(2, 2, np.mod(i, 4) + 1)
        ax.plot(epoch, res[key])
        ax.set(xlabel='MJD', ylabel=r'$m/s^2$', xlim=[epoch[0], epoch[-1]])
        ax.set_title(key)
        # ax.legend()

    plt.show()
    pass


def plot_benchmark2():
    matplotlib.rcParams.update({'font.size': 24})
    savedir = '../result/benchmarktest/'
    res = {
        'Moon': np.load(savedir + 'moon.npy'),
        'Sun': np.load(savedir + 'sun.npy'),
        'Planets': np.load(savedir + 'planets.npy'),
        'relativistic': np.load(savedir + 'relativistic.npy'),

        'EOT11a': np.load(savedir + 'eot11a.npy'),
        'FES2014b': np.load(savedir + 'fes2014.npy'),
        'Atmos tide': np.load(savedir + 'atmos.npy'),
        'dealiasing': np.load(savedir + 'AODRL06.npy'),

        'pole tide': np.load(savedir + 'earthpole.npy'),
        'ocean pole': np.load(savedir + 'oceanpole.npy'),
        'solid tide': np.load(savedir + 'solidearth.npy'),
        'Earth gravity': np.load(savedir + 'gravity.npy')
    }

    epoch = np.linspace(54650, 54651, 2880)
    plt.style.use(['grid'])
    # plt.style.use('fivethirtyeight')
    fig = plt.figure()

    ns = {}
    ds = {}
    for key in res.keys():
        ns[key] = np.linalg.norm(res[key], axis=1)
        ds[key] = ns[key].max()
    # keylist = list(res.keys())
    # for i in range(0, 4):
    #     key = keylist[i]
    #     ax = fig.add_subplot(2, 2, np.mod(i, 4) + 1)
    #     ax.plot(epoch, res[key])
    #     ax.set(xlabel='MJD', ylabel=r'$m/s^2$', xlim=[epoch[0], epoch[-1]])
    #     ax.set_title(key)
    #     # ax.legend()

    ax = fig.add_subplot(2, 1, 1)

    colors = {
        'Moon': 'cyan',
        'Sun': 'lightblue',
        'Planets': 'darkblue',
        'relativistic': 'gold',

        'FES2014b': 'grey',
        'Atmos tide': 'forestgreen',
        'dealiasing': 'red',

        'pole tide': 'lightgrey',
        'ocean pole': 'yellow',
        'solid tide': 'brown',
        'Earth gravity': 'black'
    }

    ax.axhline(y=1e-11, color='k', linestyle='--', label = r'$10^{-11} m/s^2$', linewidth = 4)

    for key in ns.keys():
        if key == 'EOT11a': continue
        ax.plot(epoch, ns[key], label=key, color = colors[key])

    ax.set(xlabel='MJD', ylabel=r'$m/s^2$', xlim=[epoch[0], epoch[-1]], ylim=[1e-26, 1e-10])
    ax.set_yscale('log')
    ax.set_title('Norm of the difference between the benchmark data and force model')
    ax.set_yticks([1e-10, 1e-12, 1e-14, 1e-16, 1e-18,1e-20, 1e-22,1e-24,1e-26])
    leg = ax.legend(bbox_to_anchor=(1, 1), loc=0, frameon=False)

    for line in leg.get_lines():
        line.set_linewidth(4.0)

    ax = fig.add_subplot(2, 1, 2)
    ds.pop('EOT11a')
    namelist = ['Earth gravity', 'Moon', 'Sun', 'Planets',
                'solid tide', 'FES2014b', 'relativistic', 'dealiasing', 'pole tide', 'Atmos tide', 'ocean pole']
    vv = [ds[k] for k in namelist]
    bernese = [2e-15, 3e-16, 0.8e-16, 0.8e-17, 2e-13, 1.5e-13, 1e-20, 3e-14, 4e-15, 6e-19, 0.9e-15]

    ax.scatter(np.arange(11), vv, marker = '*', s = 64, c = 'r', label = 'PyHawk')
    ax.scatter(np.arange(11), bernese, marker='v', c='darkblue', label='Bernese', s = 64)

    ax.set(ylabel=r'$m/s^2$', ylim=[1e-24, 1e-10])
    ax.set_yscale('log')
    ax.axhline(y=1e-11, color='k', linestyle='--', label = r'$10^{-11} m/s^2$', linewidth = 4)
    leg = ax.legend(bbox_to_anchor=(1.05, 1.0), loc=2, frameon=False)
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels(namelist, rotation = 45)


    plt.show()
    pass


def pygmt():

    pass


if __name__ == '__main__':
    # demo1()
    # demo2()
    # plot()
    # plot_benchmark2()
    pygmt()
