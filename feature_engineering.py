#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 12/3/2020 下午8:49
@Author  : Icy Huang
@Site    : 
@File    : feature_engineering.py
@Software: PyCharm Community Edition
@Python  : 
"""

import pandas as pd


def load_data():
    # https://lab.isaaclin.cn/nCoV/
    global data
    data = pd.read_csv('data/DXYArea_onlynum.csv')  # 自动把第一行作为列属性
    # 填充空数据
    data['city_zipCode'] = data['city_zipCode'].fillna('400000')
    # 转换字符串为数值
    data = data.astype(int)


def standardization():
    # 标准化，返回值为标准化后的数据
    from sklearn.preprocessing import StandardScaler

    standardScaler = StandardScaler().fit(data)
    standardScaler.transform(data)


def normalization_a():
    from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
    # 区间缩放，返回值为缩放到[0, 1]区间的数据
    minMaxScaler = MinMaxScaler().fit(data)
    minMaxScaler.transform(data)

    MaxAbsScaler = MaxAbsScaler().fit(data)
    MaxAbsScaler.transform(data)


def normalization_b():
    from sklearn.preprocessing import Normalizer
    # 正则化，返回值为归一化后的数据
    normalizer = Normalizer(norm='l2').fit(data)
    normalizer.transform(data)

    normalizer = Normalizer(norm='l1').fit(data)
    normalizer.transform(data)


if __name__ == '__main__':
    load_data()
    standardization()
    normalization_a()
    normalization_b()
