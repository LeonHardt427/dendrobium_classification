#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/22 下午6:05
# @Author  : LeonHardt
# @File    : simple_prediction.py

"""
Simple prediciton in paper 4
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# -----------------------------
# preprocessing
# -----------------------------

X = np.loadtxt('x_time_sample_F1000toT338.csv', delimiter=',')
y = np.loadtxt('y_time_label_F1000toT338.csv', delimiter=',')

sc = StandardScaler()
X = sc.fit_transform(X)

# ------------------------------
# prediction
# ------------------------------

s_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
RF_result = []
SVM_result = []
for index, (train, test) in enumerate(s_folder.split(X, y)):
    print('the {} time'.format(index))
    x_train, x_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    lda = LinearDiscriminantAnalysis(n_components=5)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)

    rf = RandomForestClassifier(n_estimators=500, criterion='entropy')
    svm = SVC(C=60.0, gamma=0.001, probability=True)

    rf.fit(x_train_lda, y_train)
    svm.fit(x_train_lda, y_train)

    rf_prediction = rf.predict(x_test_lda)
    svm_prediciton = svm.predict(x_test_lda)

    rf_accuracy = accuracy_score(y_test, rf_prediction)
    svm_accuracy = accuracy_score(y_test, svm_prediciton)

    print("RF is {}".format(rf_accuracy))
    print("SVM is {}".format(svm_accuracy))

    RF_result.append(rf_accuracy)
    SVM_result.append(svm_accuracy)

RF_acc = np.mean(RF_result)
SVM_acc = np.mean(SVM_result)

print("RF_mean is {}".format(RF_acc))
print("SVM_mean is {}".format(SVM_acc))



