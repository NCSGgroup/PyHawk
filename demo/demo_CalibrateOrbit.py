# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/6/24 14:28
# @File    : demo_CalibrateOrbit.py
# @Software: PyCharm
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.Solver.CalibrateOrbit import CalibrateOrbit


def demo1(setting_directory=None):
    if setting_directory is not None:
        setting_directory = str(Path(setting_directory).resolve())
    os.chdir(Path(__file__).resolve().parent)
    calibrate_orbit = CalibrateOrbit()
    calibrate_orbit.loadJson(setting_directory=setting_directory)
    calibrate_orbit.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--setting-dir',
        help='Directory containing the Calibrate JSON settings'
    )
    arguments = parser.parse_args()
    demo1(setting_directory=arguments.setting_dir)

