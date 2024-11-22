# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/3/25 21:54
# @File    : demo_BenchMark.py
# @Software: PyCharm
import sys
sys.path.append("../")
from src.benchmark.BenchMarkForJson import BenchMark
from src.Preference.Pre_ForceModel import ForceModelConfig
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib


config = {
    "font.family": 'serif',
    "font.size": 22,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
matplotlib.rcParams.update(config)


def demo2():
    FMConfig = ForceModelConfig()
    benchMark = BenchMark()
    '''sun'''
    print('Calculating sun...')
    threeBody_sun = json.load(open('../setting/demo_2/BenchMark_Sun.json', 'r'))
    FMConfig.__dict__ = threeBody_sun
    benchMark.three_body_sun(FMConfig)
    '''moon'''
    print('Calculating moon...')
    threeBody_moon = json.load(open('../setting/demo_2/BenchMark_Moon.json', 'r'))
    FMConfig.__dict__ = threeBody_moon
    benchMark.three_body_moon(FMConfig)
    '''three body'''
    print('Calculating planets...')
    threeBody = json.load(open('../setting/demo_2/BenchMark_ThreeBody.json', 'r'))
    FMConfig.__dict__ = threeBody
    benchMark.three_body(FMConfig)
    '''AOD'''
    print('Calculating aod...')
    aod = json.load(open('../setting/demo_2/BenchMark_AOD.json', 'r'))
    FMConfig.__dict__ = aod
    benchMark.aod(FMConfig)
    '''relativistic'''
    print('Calculating relativistic...')
    relativistic = json.load(open('../setting/demo_2/BenchMark_Relativity.json', 'r'))
    FMConfig.__dict__ = relativistic
    benchMark.relativistic(FMConfig)
    '''eot11a'''
    print('Calculating eot11a...')
    eot11a = json.load(open('../setting/demo_2/BenchMark_EOT11a.json', 'r'))
    FMConfig.__dict__ = eot11a
    benchMark.eot11a(FMConfig)
    '''FES2014'''
    print('Calculating fes2014...')
    fes2014 = json.load(open('../setting/demo_2/BenchMark_FES2104.json', 'r'))
    FMConfig.__dict__ = fes2014
    benchMark.fes2014b(FMConfig)
    '''Atmos tide'''
    print('Calculating atmos...')
    atmos = json.load(open('../setting/demo_2/BenchMark_Atmos.json', 'r'))
    FMConfig.__dict__ = atmos
    benchMark.atmos_tide(FMConfig)
    '''gravity field'''
    print('Calculating gravity field...')
    gravity_field = json.load(open('../setting/demo_2/BenchMark_Gravity.json', 'r'))
    FMConfig.__dict__ = gravity_field
    benchMark.gravity_field(FMConfig)
    '''solidEarth tide'''
    print('Calculating solid earth tide...')
    solid_earth = json.load(open('../setting/demo_2/BenchMark_SolidEarth.json', 'r'))
    FMConfig.__dict__ = solid_earth
    benchMark.solid_earth(FMConfig)
    '''solidEarth pole tide'''
    print('Calculating solid earth pole tide...')
    earth_pole = json.load(open('../setting/demo_2/BenchMark_EarthPole.json', 'r'))
    FMConfig.__dict__ = earth_pole
    benchMark.earth_pole(FMConfig)
    '''ocean pole tide'''
    print('Calculating ocean pole tide...')
    ocean_pole = json.load(open('../setting/demo_2/BenchMark_OceanPole.json', 'r'))
    FMConfig.__dict__ = ocean_pole
    benchMark.ocean_pole(FMConfig)
    pass


def plot_show():
    # matplotlib.rcParams.update({'font.size': 27})
    savedir = '../result/benchmarktest/data/'
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
    fig = plt.figure(figsize=(19, 9.6))
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
    plt.savefig(savedir + '../img/benchmark.png')
    # plt.show()

    pass


if __name__ == '__main__':
    demo2()
    plot_show()