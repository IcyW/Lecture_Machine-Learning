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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def data_preprocess():
    # https://lab.isaaclin.cn/nCoV/
    global data
    data = pd.read_csv('DXYArea_onlynum.csv', sep=',', header='infer')
    # 数据预处理
    data['city_zipCode'] = data['city_zipCode'].fillna('400000')

    # data_inf = np.isinf(data)
    # data[data_inf] = 0
    # data_nan = np.isnan(data)
    # data[data_nan] = 0
    # data.to_csv("Area_fill.csv")


def data_split():
    data_rows_len = data.shape[0]
    amount = int(0.9 * data_rows_len)

    train_data = data.head(amount)
    test_data = data.tail(data_rows_len - amount)

    global X_train, Y_train, X_test, Y_test
    X_train = train_data.iloc[:, 0:-1]
    Y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, 0:-1]
    Y_test = test_data.iloc[:, -1]


def train(k):
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    global knn_regressor
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train, Y_train)


def validate():
    global y_train_predict
    y_train_predict = knn_regressor.predict(X_train)


def test():
    global y_test_predict
    y_test_predict = knn_regressor.predict(X_test)


def verify(pred, true):
    mean_err = mean_squared_error(y_true=true, y_pred=pred)
    mean_abs_err = mean_absolute_error(y_true=true, y_pred=pred)
    r2_err = r2_score(y_true=true, y_pred=pred)
    print("mean_squared_error: %f, mean_abs_err: %f, r2_error: %f"
          % (mean_err, mean_abs_err, r2_err))


if __name__ == '__main__':
    data_preprocess()
    data_split()
    train(10)
    validate()
    test()
    print("训练集上预测结果：")
    verify(y_train_predict, Y_train)
    print("测试集上预测结果：")
    verify(y_test_predict, Y_test)
