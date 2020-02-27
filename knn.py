#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 25/2/2020 下午9:13
@Author  : Icy Huang
@Site    : 
@File    : knn.py
@Software: PyCharm Community Edition
@Python  : 
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def data_preprocess():
    # https://lab.isaaclin.cn/nCoV/
    global data
    data = pd.read_csv('DXYArea_onlynum.csv', sep=',', header='infer')
    # 数据预处理
    data.fillna(400000)  # 数据里只有city_zipCode缺失，填充一下
    data_inf = np.isinf(data)
    data[data_inf] = 0
    data_nan = np.isnan(data)
    data[data_nan] = 0
    # data.to_csv("Area_fill")


def data_split():
    amount = int(0.9 * data.size)

    train_data = data.head(amount)
    test_data = data.tail(data.size - amount)

    global X_train, Y_train, X_test, Y_test
    X_train = train_data.iloc[:, 0:-1]
    Y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, 0:-1]
    Y_test = test_data.iloc[:, -1]


def train():
    global knn_regressor
    knn_regressor = KNeighborsRegressor(n_neighbors=10)
    knn_regressor.fit(X_train, Y_train)


def test():
    global y_predict
    y_predict = knn_regressor.predict(X_test)


def verify():
    mean_err = mean_squared_error(y_true=Y_test, y_pred=y_predict)
    r2_err = r2_score(y_true=Y_test, y_pred=y_predict)
    print("mean_squared_error: %f, r2_error: %f" % (mean_err, r2_err))


if __name__ == '__main__':
    data_preprocess()
    data_split()
    train()
    test()
    verify()
