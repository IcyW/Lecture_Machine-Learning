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
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from plot_curves import plot_pr, plot_roc
from handwritten.KNNregressor import KNNregressor
from handwritten.KNNclassifier import KNNclassifier


def data_preprocess():
    # https://lab.isaaclin.cn/nCoV/
    global data
    data = pd.read_csv('data/DXYArea_onlynum.csv')  # 自动把第一行作为列属性
    # 填充空数据
    data['city_zipCode'] = data['city_zipCode'].fillna('400000')
    # 转换字符串为数值
    data = data.astype(int)

    # data.to_csv("Area_fill_int.csv")


def shuffle_data():
    global shuffled_data
    # data = pd.read_csv('data/Area_fill_int.csv')
    shuffled_data = shuffle(data)

    # shuffled_data.to_csv('data/Area_fill_shuffled.csv')


def binarize_data():
    # shuffled_data = pd.read_csv('data/Area_fill_shuffled.csv')
    shuffled_data.loc[shuffled_data.city_deadCount > 0, 'city_deadCount'] = 1
    # 相当于以下逻辑
    # Y_data = shuffled_data.iloc[:, -1]
    # for i in range(0, len(Y_data)):
    #     shuffled_data[i, 'city_deadCount'] = 1 if Y_data[i] > 0 else 0

    # shuffled_data.to_csv('data/Area_fill_shuffled_binarized.csv')


def data_split():
    # final_data = pd.read_csv('data/Area_fill_shuffled.csv')
    # final_data = pd.read_csv('data/Area_fill_shuffled_binarized.csv')
    final_data = shuffled_data
    data_rows_len = final_data.shape[0]
    train_amount = int(0.003 * data_rows_len)
    # test_amount = data_rows_len - train_amount
    test_amount = int(0.001 * data_rows_len)

    train_data = final_data.head(train_amount)
    test_data = final_data.tail(test_amount)

    global X_train, Y_train, X_test, Y_test
    X_train = train_data.iloc[:, 0:-1]
    Y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, 0:-1]
    Y_test = test_data.iloc[:, -1]
    print("训练集样本量： %d, 测试集样本量： %d" % (train_amount, test_amount))


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


def knn_handwritten(k):
    global knn_regressor_hand, y_train_predict_hand, y_test_predict_hand
    knn_regressor_hand = KNNregressor(k=k)
    knn_regressor_hand.fit(X_train=X_train, y_train=Y_train)
    y_train_predict_hand = knn_regressor_hand.predict(X_train)
    y_test_predict_hand = knn_regressor_hand.predict(X_test)


def verify(pred, true):
    mean_err = mean_squared_error(y_true=true, y_pred=pred)
    mean_abs_err = mean_absolute_error(y_true=true, y_pred=pred)
    r2_err = r2_score(y_true=true, y_pred=pred)
    print("mean_squared_error: %f, mean_abs_err: %f, r2_error: %f"
          % (mean_err, mean_abs_err, r2_err))


def knn_handwritten_classifier(k):
    global knn_classifier_hand, y_train_predict_hand_c, y_test_predict_hand_c
    knn_classifier_hand = KNNclassifier(k=k)
    knn_classifier_hand.fit(X_train=X_train, y_train=Y_train)
    y_train_predict_hand_c = knn_classifier_hand.predict(X_train)
    y_test_predict_hand_c = knn_classifier_hand.predict(X_test)


def regressor():
    # step1. 数据预处理
    data_preprocess()
    shuffle_data()
    data_split()
    # step2. knn调包
    start = datetime.now()
    train(5)
    validate()
    test()
    print("训练集上预测结果：")
    verify(y_train_predict, Y_train)
    print("测试集上预测结果：")
    verify(y_test_predict, Y_test)
    halt = datetime.now()
    print("耗时：%0.1f 毫秒" % (halt - start).microseconds)

    # step3. knn手写
    knn_handwritten(5)
    print("训练集上预测结果（手写knn）：")
    verify(y_train_predict_hand, Y_train)
    print("测试集上预测结果（手写knn）：")
    verify(y_test_predict_hand, Y_test)

    end = datetime.now()
    print("耗时：%f 秒" % (end - halt).seconds)


def classifier():
    # step1. 数据预处理
    data_preprocess()
    shuffle_data()
    binarize_data()  # 二分类
    data_split()

    # step3. knn手写
    knn_handwritten_classifier(5)


if __name__ == '__main__':
    # regressor()
    classifier()
