# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/4/24 22:46
# @File    : Pre_Accelerometer.py
# @Software: PyCharm
import json


class AccelerometerConfig:
    def __init__(self):
        self.cali = {'Scale': [False, False, False,
                               False, False, False,
                               False, False, False],
                     'Bias_constant': [True, True, True],
                     'Bias_trend': [True, True, True],
                     'Bias_quadratic': [False, False, False]
                     }
        self.cali_value = {'Scale': [0.9595, 0, 0,
                                     0, 0.9797, 0,
                                     0, 0, 0.9485],
                           'Bias_constant': [0, 0, 0],
                           'Bias_trend': [0, 0, 0],
                           'Bias_quadratic': [0, 0, 0]
                           }
        self.temp_non_conservative_data = "../temp/NonConservativeForce"


def demo2():
    Obj1 = AccelerometerConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/AccelerometerConfig_A.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    demo2()

