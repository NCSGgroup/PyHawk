# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/6/24 14:28
# @File    : demo_CalibrateOrbit.py
# @Software: PyCharm
import sys
sys.path.append("../")
from src.Solver.CalibrateOrbit import CalibrateOrbit


def main():
    calibrate_orbit = CalibrateOrbit()
    calibrate_orbit.loadJson()
    calibrate_orbit.run()


if __name__ == '__main__':
    main()

