# -*- coding: utf-8 -*-
# @Author  : Wuyi
# @Time    : 2023/7/17 14:36
# @File    : demo_test_FormatWrite.py
# @Software: PyCharm
from src.Auxilary.Format import FormatWrite


def demo1():
    data_span = ['2008-01-01', '2008-01-31']
    c = [1,2,3,4,5]
    s = [1,2,3,4,5]
    sigmac = [1,2,3,4,5]
    sigmas = [1,2,3,4,5]
    format = FormatWrite().configure(filedir='../result/SH', data=data_span, degree=60, c=c, s=s, sigmaS=sigmas, sigmaC=sigmac)
    format.unfiltered()


if __name__ == '__main__':
    demo1()
