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


def binning_equal_freq():
    # 新增一列存储等频划分的分箱特征, pd.qcut(属性, 被分为几组)
    data['province_confirmedCount_bin1'] = pd.qcut(data['province_confirmedCount'], 5)
    print(data['province_confirmedCount_bin1'])
    print(data['province_confirmedCount_bin1'].value_counts())

    # (526.0, 1073.0]      11575
    # (0.999, 119.0]       11568
    # (119.0, 283.0]       11550
    # (283.0, 526.0]       11512
    # (1073.0, 65596.0]    11489


def binning_equal_interval():
    # 新增一列存储等距划分的分箱特征, pd.cut(属性, 被分为几组)
    data['province_confirmedCount_bin1'] = pd.cut(data['province_confirmedCount'], 10)
    data['province_confirmedCount_bin2'] = pd.cut(data['province_confirmedCount'],
                                                  bins=[-65, 13120, 26239, 39358, 52477, 65596])
    # print(data['province_confirmedCount_bin1'])
    print(data['province_confirmedCount_bin2'])
    print(data['province_confirmedCount_bin2'].value_counts())

    # (-65, 13120]      54076
    # (52477, 65596]     1591
    # (13120, 26239]     1262
    # (26239, 39358]      544
    # (39358, 52477]      221


def binning_clustering():
    from sklearn.cluster import KMeans
    k = 5
    # attribute = data.loc[:, 'province_confirmedCount': 'province_curedCount']
    attribute = data['province_confirmedCount']
    kmodel = KMeans(n_clusters=k)  # k为聚成几类
    kmodel.fit(attribute.values.reshape(len(attribute), 1))  # 训练模型
    c = pd.DataFrame(kmodel.cluster_centers_)  # 求聚类中心
    # print(c)
    c = c.sort_values(by=0)  # 排序, 在聚类过程中需要保证分箱的有序性　　
    mean = c.rolling(2).mean()  # Size of the moving window
    # print(mean)
    w = mean.iloc[1:]  # 用滑动窗口求均值的方法求相邻两项求中点，作为边界点
    # print("w:")
    w = [0] + list(w[0]) + [attribute.max()]  # 把首末边界点加上, 并转换为list格式
    # print(w)
    data['province_confirmedCount_bin3'] = pd.cut(attribute, w)  # cut函数, labels=range(k-1)
    print(data['province_confirmedCount_bin3'].value_counts())


def binarization():
    from sklearn.preprocessing import Binarizer
    # Binarizer函数也可以设定一个阈值，结果数据值大于阈值的为1，小于阈值的为0
    binarizer = Binarizer(threshold=0.0).fit(data)
    binarizer.transform(data)


if __name__ == '__main__':
    load_data()
    # standardization()
    # normalization_a()
    # normalization_b()
    # binning_equal_freq()
    # binning_equal_interval()
    binning_clustering()









