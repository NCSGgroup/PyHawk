# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/4/22 22:50
# @File    : demo_test_CarTopy.py
# @Software: PyCharm
import cartopy
import matplotlib.pyplot as plt  ###引入库包
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import cartopy.crs as ccrs
import numpy as np


if __name__ == '__main__':
    # getKBRRGMT()
    res = np.load("PreKBRR.npz", allow_pickle=True)
    x = np.degrees(np.array(res['x']))
    y = np.degrees(np.array(res['y']))
    z = res['z']

    proj = ccrs.PlateCarree()
    x_extent = [-180,-120,-60,0,60,120,180]  # 经纬度范围,#直接使用(-180,-120,-60,0,60,120,180)会异常，需要写成(0, 60, 120, 180, 240, 300, 360)的形式
    y_extent = [-90, -60, -30, 0, 30, 60, 90]
    fig = plt.figure(figsize=(4, 4), dpi=200)  # 创建画布
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))  # 创建子图
    ax.coastlines()  # 调用ax的方法画海岸线
    # clist = ['red', 'lime', 'blue']
    # newcmp = LinearSegmentedColormap.from_list('chaos', clist)
    l = ax.scatter(x, y, c=z, s=1, cmap=plt.cm.Spectral, vmax=1e-5, vmin=-1e-5)
    ax.set_xticks(x_extent, crs=ccrs.PlateCarree())  # 添加经纬度
    ax.set_yticks(y_extent, crs=ccrs.PlateCarree())
    fig.colorbar(mappable=l)
    plt.show()

