# -*- coding: utf-8 -*-
# @Time    : 2017/12/27 9:35
# @Author  : LeonHardt
# @File    : icp_classification.py

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory
from nonconformist.evaluation import class_avg_c, class_mean_errors

# ---------------------------------------------
# intialize
# ---------------------------------------------
X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

sc = StandardScaler()
X = sc.fit_transform(X)

# -------------------------------------------------
# Kfold test
# -------------------------------------------------

s_folder = StratifiedKFold(n_splits=10, shuffle=True)
for index, (train, test) in enumerate(s_folder.split(X, y)):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(X_train, y_train, train_size=0.7, shuffle=True)
    y_test = y_test.reshape((-1, 1))

    model = SVC(kernel='rbf', C=4000, gamma=0.001, probability=True)
    nc = NcFactory.create_nc(model=model)
    icp = IcpClassifier(nc)

    icp.fit(x_train_sp, y_train_sp)
    icp.calibrate(x_cal, y_cal)
    prediction = icp.predict(X_test, significance=None)

    result = [index, class_mean_errors(prediction, y_test, significance=0.5),
              class_avg_c(prediction, y_test, significance=0.5)]
    if index == 0:
        result_summary = result
    else:
        result_summary = np.vstack((result_summary, result))

df_summary = pd.DataFrame(result_summary, columns=['index', 'mean_errors', 'average_count'])
df_summary.to_csv('icp_svm')
print(df_summary)


