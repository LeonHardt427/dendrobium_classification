# -*- coding: utf-8 -*-
# @Time    : 2017/12/27 9:35
# @Author  : LeonHardt
# @File    : icp_classification.py

import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory
from nonconformist.evaluation import class_avg_c, class_mean_errors

# ---------------------------------------------
# intialize
# ---------------------------------------------


path = os.getcwd()

X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

sc = StandardScaler()
X = sc.fit_transform(X)

# -------------------------------------------------
# prediction: k_fold
# -------------------------------------------------
# -------------------------
# select_model
# -------------------------

model = SVC(kernel='rbf', C=4000, gamma=0.001, probability=True)
model_name = 'SVM'

# model = RandomForestClassifier(n_estimators=12, criterion='entropy')
# model_name = 'RandomForest'

# model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
# model_name = 'Tree'

# model = KNeighborsClassifier(n_neighbors=3)
# model_name = '3NN'

# model = KNeighborsClassifier(n_neighbors=1)
# model_name = '1NN'
# -------------------------

significance = 0.5
result_summary = []

s_folder = StratifiedKFold(n_splits=10, shuffle=True)
for index, (train, test) in enumerate(s_folder.split(X, y)):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(X_train, y_train, test_size=0.5, shuffle=True)
    y_test = y_test.reshape((-1, 1))

    nc = NcFactory.create_nc(model=model)
    icp = IcpClassifier(nc)

    icp.fit(x_train_sp, y_train_sp)
    icp.calibrate(x_cal, y_cal)
    prediction = icp.predict(X_test, significance=None)

    result = [1-class_mean_errors(prediction, y_test, significance=significance),
              class_avg_c(prediction, y_test, significance=significance)]
    if index == 0:
        result_summary = result
    else:
        result_summary = np.vstack((result_summary, result))
    print('\nICP')
    print('Accuracy: {}'.format(result[0]))
    print('Average count: {}'.format(result[1]))

df_summary = pd.DataFrame(result_summary, columns=['Accuracy', 'Average_count'])

summary_path = path + '/summary/icp/'
summary_file = summary_path + 'icp_' + model_name + '.csv'
if os.path.exists(summary_path) is not True:
    os.makedirs(summary_path)
if os.path.exists(summary_file) is True:
    os.remove(summary_file)

df_summary.to_csv(summary_file)
print(df_summary)



