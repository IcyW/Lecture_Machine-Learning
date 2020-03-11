#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 9/3/2020 下午9:48
@Author  : Icy Huang
@Site    : 
@File    : plot_curves.py
@Software: PyCharm Community Edition
@Python  : 
"""

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def plot_roc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUC_score = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic(area = %0.2f)' % AUC_score)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(fpr, tpr, marker='o')
    plt.show()


def plot_pr(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.title('P-R Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    plt.show()
