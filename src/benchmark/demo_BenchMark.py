# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:54
# @File    : demo_BenchMark.py
# @Software: PyCharm
import pygmt
from src.benchmark.BenchMarkForJson import BenchMark
from src.Preference.Pre_ForceModel import ForceModelConfig
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import json
import matplotlib
# import scienceplots

config = {
    "font.family": 'serif',
    "font.size": 22,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
matplotlib.rcParams.update(config)

def demo1():
    FMConfig = ForceModelConfig()
    benchMark = BenchMark()
    '''three body'''
    threeBody = json.load(open('../../setting/demo_2/BenchMark_ThreeBody.json', 'r'))
    FMConfig.__dict__ = threeBody
    benchMark.three_body_sun(FMConfig)
    '''AOD'''
    aod = json.load(open('../../setting/demo_2/BenchMark_AOD.json', 'r'))
    FMConfig.__dict__ = aod
    benchMark.aod(FMConfig)
    '''relativistic'''
    relativistic = json.load(open('../../setting/demo_2/BenchMark_Relativity.json', 'r'))
    FMConfig.__dict__ = relativistic
    benchMark.relativistic(FMConfig)
    '''eot11a'''
    eot11a = json.load(open('../../setting/demo_2/BenchMark_EOT11a.json', 'r'))
    FMConfig.__dict__ = eot11a
    benchMark.eot11a(FMConfig)
    '''FES2014'''
    fes2014 = json.load(open('../../setting/demo_2/BenchMark_FES2104.json', 'r'))
    FMConfig.__dict__ = fes2014
    benchMark.fes2014b(FMConfig)
    '''Atmos tide'''
    atmos = json.load(open('../../setting/demo_2/BenchMark_Atmos.json', 'r'))
    FMConfig.__dict__ = atmos
    benchMark.atmos_tide(FMConfig)
    '''gravity field'''
    gravity_field = json.load(open('../../setting/demo_2/BenchMark_Gravity.json', 'r'))
    FMConfig.__dict__ = gravity_field
    benchMark.gravity_field(FMConfig)
    '''solidEarth tide'''
    solid_earth = json.load(open('../../setting/demo_2/BenchMark_SolidEarth.json', 'r'))
    FMConfig.__dict__ = solid_earth
    benchMark.solid_earth(FMConfig)
    '''solidEarth pole tide'''
    earth_pole = json.load(open('../../setting/demo_2/BenchMark_EarthPole.json', 'r'))
    FMConfig.__dict__ = earth_pole
    benchMark.earth_pole(FMConfig)
    '''ocean pole tide'''
    ocean_pole = json.load(open('../../setting/demo_2/BenchMark_OceanPole.json', 'r'))
    FMConfig.__dict__ = ocean_pole
    benchMark.ocean_pole(FMConfig)
    pass


def demo2():
    diff = np.load('../../result/benchmarktest/data/sun.npy')
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
    fig.subplots_adjust(left=0.085, right=0.817, top=0.932, bottom=0.191,
                        wspace=0.42, hspace=0.365)
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
    ax.text(-0.1, 1.05, 'a)', fontsize=2, transform=ax.transAxes)
    ax.set(xlabel='MJD', ylabel=r'$m/s^2$', xlim=[epoch[0], epoch[-1]], ylim=[1e-26, 1e-10])
    ax.set_yscale('log')
    ax.set_yticks([1e-10, 1e-14, 1e-18, 1e-22, 1e-26])
    leg = ax.legend(bbox_to_anchor=(1, 1.3), loc=0, frameon=False)

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
    ax.text(-0.1, 1.05, 'b)', fontsize=24, transform=ax.transAxes)
    ax.set(ylabel=r'$m/s^2$', ylim=[1e-24, 1e-10])
    ax.set_yscale('log')
    ax.axhline(y=1e-11, color='k', linestyle='--', label = r'$10^{-11} m/s^2$', linewidth = 4)
    leg = ax.legend(bbox_to_anchor=(1, 0.5), loc=0, frameon=False)
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels(namelist, rotation = 45)


    plt.show()
    pass


def pygmt_test():
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

    ns = {}
    ds = {}
    for key in res.keys():
        ns[key] = np.linalg.norm(res[key], axis=1)
        ds[key] = ns[key].max()

    epoch = np.linspace(54650, 54651, 2880)

    fig = pygmt.Figure()
    fig.plot(
        region=[54650, 54651, 10e-12, 10e-10],
        projection="X15c/10c",
        frame=["a"],
        x=epoch,
        y=ns['Moon'],
        pen="2p, red",
    )

    fig.show()


    pass


def plot_benchmark3():
    # matplotlib.rcParams.update({'font.size': 27})
    savedir = '../result/benchmarktest/'
    res = {
        'Moon': np.load(savedir + 'moon.npy'),
        'Sun': np.load(savedir + 'sun.npy'),
        'Planets': np.load(savedir + 'planets.npy'),
        'Relativistic': np.load(savedir + 'relativistic.npy'),

        'EOT11a': np.load(savedir + 'eot11a.npy'),
        'FES2014b': np.load(savedir + 'fes2014.npy'),
        'Atmos tide': np.load(savedir + 'atmos.npy'),
        'Dealiasing': np.load(savedir + 'AODRL06.npy'),

        'Pole tide': np.load(savedir + 'earthpole.npy'),
        'Ocean pole': np.load(savedir + 'oceanpole.npy'),
        'Solid tide': np.load(savedir + 'solidearth.npy'),
        'Earth gravity': np.load(savedir + 'gravity.npy')
    }

    epoch = np.linspace(54650, 54651, 2880)
    # plt.style.use(['science'])
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([0.11, 0.11, 0.8, 0.7])

    # ax.tick_params(axis="both", which="major", direction="in", width=1, length=5)
    # ax.tick_params(axis="both", which="minor", direction="in", width=1, length=3)
    # ax.xaxis.set_minor_locator(MultipleLocator(0.4))

    border = plt.gca()
    border.spines['top'].set_linewidth(2)  # 设置顶部边框线宽度为2
    border.spines['right'].set_linewidth(2)  # 设置右侧边框线宽度为2
    border.spines['bottom'].set_linewidth(2)  # 设置底部边框线宽度为2
    border.spines['left'].set_linewidth(2)  # 设置左侧边框线宽度为2
    # border.spines['left'].set_linestyle('-.')
    # border.spines['bottom'].set_linestyle('-.')
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
    # colors = {
    #     'Moon': 'cyan',
    #     'Sun': 'lightblue',
    #     'Planets': 'darkblue',
    #     'relativistic': 'gold',
    #
    #     'FES2014b': 'grey',
    #     'Atmos tide': 'forestgreen',
    #     'dealiasing': 'red',
    #
    #     'pole tide': 'lightgrey',
    #     'ocean pole': 'yellow',
    #     'solid tide': 'brown',
    #     'Earth gravity': 'black'
    # }
    colors = {
        'Moon': '#2c2c54',
        'Sun': '#474787',
        'Planets': '#aaa69d',
        'Relativistic': '#227093',
        'FES2014b': '#218c74',
        'Atmos tide': '#b33939',
        'Dealiasing': '#cd6133',
        'Pole tide': '#FC427B',
        'Ocean pole': '#0097e6',
        'Solid tide': '#BDC581',
        'Earth gravity': '#F3B169'
    }
    ax.axhline(y=1e-11, color='#e84118', linestyle='--', label = 'Baseline', linewidth = 5)
    ax.text(54650.85, 5e-13, r'$10^{-11} \mathrm{m/s^2}$', c='k')
    for key in ns.keys():
        if key == 'EOT11a': continue
        plt.plot(epoch, ns[key], label=key, color = colors[key], linewidth=3)
    # ax.text(-0.1, 1.05, 'a)', fontsize=2, transform=ax.transAxes)
    ax.set(xlabel='MJD', ylabel=r'$\mathrm{m/s^2}$', xlim=[epoch[0], epoch[-1]], ylim=[1e-26, 1e-10])
    ax.set_yscale('log')

    ax.set_yticks([1e-10, 1e-14, 1e-18, 1e-22, 1e-26])
    ax.set_xticks([54650.0, 54650.2, 54650.4, 54650.6, 54650.8, 54651.0], ['54650.0', '54650.2', '54650.4', '54650.6', '54650.8', '54651.0'])
    leg = ax.legend(ncol=4, bbox_to_anchor=(0.5, 1.25), loc='upper center', borderpad=0, labelspacing=0.2, columnspacing=0.7,frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.minorticks_on()
    ax.grid(True, linestyle="--")
    # plt.grid(True, which="minor", linestyle=":", color="lightgray", linewidth=0.75)
    plt.show()
    pass


def plot_benchmark4():
    savedir = '../result/benchmarktest/'
    res = {
        'Moon': np.load(savedir + 'moon.npy'),
        'Sun': np.load(savedir + 'sun.npy'),
        'Planets': np.load(savedir + 'planets.npy'),
        'Relativistic': np.load(savedir + 'relativistic.npy'),

        'EOT11a': np.load(savedir + 'eot11a.npy'),
        'FES2014b': np.load(savedir + 'fes2014.npy'),
        'Atmos tide': np.load(savedir + 'atmos.npy'),
        'Dealiasing': np.load(savedir + 'AODRL06.npy'),

        'Pole tide': np.load(savedir + 'earthpole.npy'),
        'Ocean pole': np.load(savedir + 'oceanpole.npy'),
        'Solid tide': np.load(savedir + 'solidearth.npy'),
        'Earth gravity': np.load(savedir + 'gravity.npy')
    }

    epoch = np.linspace(54650, 54651, 2880)
    # plt.style.use(['science'])
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([0.11, 0.21, 0.82, 0.72])

    # ax.tick_params(axis="both", which="major", direction="in", width=1, length=5)
    # ax.tick_params(axis="both", which="minor", direction="in", width=1, length=3)
    # ax.xaxis.set_minor_locator(MultipleLocator(0.4))

    border = plt.gca()
    border.spines['top'].set_linewidth(2)  # 设置顶部边框线宽度为2
    border.spines['right'].set_linewidth(2)  # 设置右侧边框线宽度为2
    border.spines['bottom'].set_linewidth(2)  # 设置底部边框线宽度为2
    border.spines['left'].set_linewidth(2)  # 设置左侧边框线宽度为2
    # border.spines['left'].set_linestyle('-.')
    # border.spines['bottom'].set_linestyle('-.')
    ns = {}
    ds = {}
    for key in res.keys():
        ns[key] = np.linalg.norm(res[key], axis=1)
        ds[key] = ns[key].max()

    # colors = {
    #     'Moon': 'cyan',
    #     'Sun': 'lightblue',
    #     'Planets': 'darkblue',
    #     'relativistic': 'gold',
    #
    #     'FES2014b': 'grey',
    #     'Atmos tide': 'forestgreen',
    #     'dealiasing': 'red',
    #
    #     'pole tide': 'lightgrey',
    #     'ocean pole': 'yellow',
    #     'solid tide': 'brown',
    #     'Earth gravity': 'black'
    # }
    colors = {
        'Moon': '#FFC312',
        'Sun': '#C4E538',
        'Planets': '#12CBC4',
        'relativistic': '#273c75',

        'FES2014b': '#ED4C67',
        'Atmos tide': '#00a8ff',
        'dealiasing': '#9c88ff',

        'pole tide': '#fbc531',
        'ocean pole': '#4cd137',
        'solid tide': '#487eb0',
        'Earth gravity': '#2f3640'
    }

    ds.pop('EOT11a')
    namelist = ['Earth gravity', 'Moon', 'Sun', 'Planets',
                'Solid tide', 'FES2014b', 'Relativistic', 'Dealiasing', 'Pole tide', 'Atmos tide', 'Ocean pole']
    vv = [ds[k] for k in namelist]
    bernese = [2e-15, 3e-16, 0.8e-16, 0.8e-17, 2e-13, 1.5e-13, 1e-20, 3e-14, 4e-15, 6e-19, 0.9e-15]

    ax.scatter(np.arange(11), vv, marker='*', s=300, c='r', label='HAWK')
    ax.scatter(np.arange(11), bernese, marker='v', c='darkblue', label='Bernese', s=300)
    # ax.text(-0.1, 1.05, 'b)', fontsize=24, transform=ax.transAxes)
    ax.set(ylabel=r'$\mathrm{m/s^2}$', ylim=[1e-24, 1e-10])
    ax.set_yscale('log')
    ax.axhline(y=1e-11, color='k', linestyle='--', label='Baseline', linewidth=5)
    ax.text(9, 5e-13, r'$10^{-11} \mathrm{m/s^2}$', c='k')

    leg = ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.13), loc='upper center',  frameon=False)
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels(namelist, rotation=45)
    ax.grid(True, linestyle="--")
    plt.minorticks_on()
    plt.show()
    pass


def plot_benchmark5():
    # matplotlib.rcParams.update({'font.size': 27})
    savedir = '../result/benchmarktest/'
    res = {
        'Moon': np.load(savedir + 'moon.npy'),
        'Sun': np.load(savedir + 'sun.npy'),
        'Planets': np.load(savedir + 'planets.npy'),
        'Relativistic': np.load(savedir + 'relativistic.npy'),

        'EOT11a': np.load(savedir + 'eot11a.npy'),
        'FES2014b': np.load(savedir + 'fes2014.npy'),
        'Atmos tide': np.load(savedir + 'atmos.npy'),
        'Dealiasing': np.load(savedir + 'AODRL06.npy'),

        'Pole tide': np.load(savedir + 'earthpole.npy'),
        'Ocean pole': np.load(savedir + 'oceanpole.npy'),
        'Solid tide': np.load(savedir + 'solidearth.npy'),
        'Earth gravity': np.load(savedir + 'gravity.npy')
    }

    epoch = np.linspace(54650, 54651, 2880)
    # plt.style.use(['science'])
    # fig = plt.figure(figsize=(15, 9))
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    # ax1 = fig.add_axes([0.11, 0.11, 0.8, 0.7])

    # ax.tick_params(axis="both", which="major", direction="in", width=1, length=5)
    # ax.tick_params(axis="both", which="minor", direction="in", width=1, length=3)
    # ax.xaxis.set_minor_locator(MultipleLocator(0.4))

    border = plt.gca()
    border.spines['top'].set_linewidth(2)  # 设置顶部边框线宽度为2
    border.spines['right'].set_linewidth(2)  # 设置右侧边框线宽度为2
    border.spines['bottom'].set_linewidth(2)  # 设置底部边框线宽度为2
    border.spines['left'].set_linewidth(2)  # 设置左侧边框线宽度为2
    # border.spines['left'].set_linestyle('-.')
    # border.spines['bottom'].set_linestyle('-.')
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
    # colors = {
    #     'Moon': 'cyan',
    #     'Sun': 'lightblue',
    #     'Planets': 'darkblue',
    #     'relativistic': 'gold',
    #
    #     'FES2014b': 'grey',
    #     'Atmos tide': 'forestgreen',
    #     'dealiasing': 'red',
    #
    #     'pole tide': 'lightgrey',
    #     'ocean pole': 'yellow',
    #     'solid tide': 'brown',
    #     'Earth gravity': 'black'
    # }
    colors = {
        'Moon': '#2c2c54',
        'Sun': '#474787',
        'Planets': '#aaa69d',
        'Relativistic': '#227093',
        'FES2014b': '#218c74',
        'Atmos tide': '#b33939',
        'Dealiasing': '#cd6133',
        'Pole tide': '#FC427B',
        'Ocean pole': '#0097e6',
        'Solid tide': '#BDC581',
        'Earth gravity': '#F3B169'
    }

    ax1.axhline(y=1e-11, color='#e84118', linestyle='--', label='Baseline', linewidth=5)
    ax1.text(54650.85, 5e-13, r'$10^{-11} \mathrm{m/s^2}$', c='k')
    for key in ns.keys():
        if key == 'EOT11a': continue
        plt.plot(epoch, ns[key], label=key, color=colors[key], linewidth=3)
    # ax.text(-0.1, 1.05, 'a)', fontsize=2, transform=ax.transAxes)
    # ax1.set_xlabel('Hour', horizontalalignment='right', labelpad=1)
    ax1.set_xlim([epoch[0], epoch[-1]])
    ax1.set_ylabel(r'$\mathrm{m/s^2}$')
    ax1.set_ylim([1e-26, 1e-10])
    ax1.set_yscale('log')

    ax1.text(-0.05, 1.15, 'a)', fontsize=24, transform=ax1.transAxes)
    ax1.set_yticks([1e-10, 1e-14, 1e-18, 1e-22, 1e-26])
    ax1.set_xticks([54650.0, 54650.166, 54650.332, 54650.498, 54650.664, 54650.830, 54651.0],
                  ['0', '4', '8', '12', '16', '20', '24 (h)'])
    leg = ax1.legend(ncol=4, bbox_to_anchor=(0.5, 1.32), loc='upper center', borderpad=0, labelspacing=0.2,
                    columnspacing=2.0, frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.minorticks_on()
    ax1.grid(True, linestyle="--")
    # plt.grid(True, which="minor", linestyle=":", color="lightgray", linewidth=0.75)

    ## fig 2
    ax2 = fig.add_subplot(212)

    border = plt.gca()
    border.spines['top'].set_linewidth(2)  # 设置顶部边框线宽度为2
    border.spines['right'].set_linewidth(2)  # 设置右侧边框线宽度为2
    border.spines['bottom'].set_linewidth(2)  # 设置底部边框线宽度为2
    border.spines['left'].set_linewidth(2)  # 设置左侧边框线宽度为2
    # border.spines['left'].set_linestyle('-.')
    # border.spines['bottom'].set_linestyle('-.')
    ns = {}
    ds = {}
    for key in res.keys():
        ns[key] = np.linalg.norm(res[key], axis=1)
        ds[key] = ns[key].max()

    ds.pop('EOT11a')
    namelist = ['Earth gravity', 'Moon', 'Sun', 'Planets',
                'Solid tide', 'FES2014b', 'Relativistic', 'Dealiasing', 'Pole tide', 'Atmos tide', 'Ocean pole']
    vv = [ds[k] for k in namelist]
    bernese = [2e-15, 3e-16, 0.8e-16, 0.8e-17, 2e-13, 1.5e-13, 1e-20, 3e-14, 4e-15, 6e-19, 0.9e-15]

    ax2.scatter(np.arange(11), vv, marker='*', s=300, c='r', label='PyHawk')
    ax2.scatter(np.arange(11), bernese, marker='v', c='darkblue', label='Bernese', s=300)
    ax2.text(-0.05, 1.02, 'b)', fontsize=24, transform=ax2.transAxes)
    ax2.set(ylabel=r'$\mathrm{m/s^2}$', ylim=[1e-24, 1e-10])
    ax2.set_yscale('log')
    ax2.axhline(y=1e-11, color='k', linestyle='--', label='Baseline', linewidth=5)
    ax2.text(9, 5e-13, r'$10^{-11} \mathrm{m/s^2}$', c='k')

    leg = ax2.legend(ncol=3, bbox_to_anchor=(0.5, 1.17), loc='upper center', frameon=False)
    ax2.set_xticks(np.arange(11))
    ax2.set_xticklabels(namelist, rotation=25)
    ax2.grid(True, linestyle="--")
    plt.minorticks_on()

    fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1, hspace=0.3)
    plt.show()

    pass


if __name__ == '__main__':
    demo1()
    # demo2()
    # plot()
    # plot_benchmark3()
    # plot_benchmark4()
    # plot_benchmark5()
    # pygmt_test()
