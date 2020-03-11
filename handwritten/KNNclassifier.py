#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 9/3/2020 下午10:14
@Author  : Icy Huang
@Site    : 
@File    : KNNclassifier.py
@Software: PyCharm Community Edition
@Python  : 
"""

from math import sqrt
from collections import Counter
import numpy as np


class KNNclassifier:
    def __init__(self, k):
        self._k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """
            根据训练数据集X_train和y_train训练kNN分类器
        :param X_train:  训练数据集的特征
        :param y_train:  训练数据集的结果
        :return: self
        """
        self._X_train = X_train
        self._y_train = y_train
        return self

    def _eu_distance(self, x):
        """
            计算欧氏距离
        :param x: 当前样本
        :return: distances: 记录x到样本数据集中每个点的距离
        """
        # distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
        distances = []
        for i in range(0, len(self._X_train)):
            x_train = self._X_train.iloc[i]
            d = sqrt(np.sum((x_train - x) ** 2))
            distances.append(d)
        return distances

    def _k_nearest_vote(self, x):
        """
            计算与x最相近k个样本的对应"投票"结果: 取平均
        :param x: 当前样本
        :return: 预期结果
        """
        distances = self._eu_distance(x)

        nearest = np.argsort(distances)
        # shuffle后df.series格式的数据索引还是原样，求最近距离后索引会出错
        np_y_train = np.array(self._y_train)
        topK_y = [np_y_train[i] for i in nearest[:self._k]]
        votes = Counter(topK_y)
        (label, freq) = votes.most_common(1)[0]
        # List the n most common elements and their counts from the most common to the least.
        return label

    def predict(self, X_predict):
        """
            对数据集进行结果预测
        :param X_predict: 待预测数据集
        :return: 返回表示X_predict结果的向量
        """
        # y_predict = [self._k_nearest_vote(x) for x in X_predict]
        y_predict = []
        for i in range(0, len(X_predict)):
            x = X_predict.iloc[i]
            y_predict.append(self._k_nearest_vote(x))
        return np.array(y_predict)

    # def __repr__(self):
    #     return "kNN(k=%d)" % self._k
