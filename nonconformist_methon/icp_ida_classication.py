# -*- coding: utf-8 -*-
# @Time    : 2017/12/31 11:23
# @Author  : LeonHardt
# @File    : icp_ida_classication.py

import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory
from nonconformist.evaluation import class_avg_c, class_mean_errors
from force_value import force_mean_errors

# ---------------------------------------------
# intialize
# ---------------------------------------------

path = os.getcwd()
#
# X = np.loadtxt('x_sample.csv', delimiter=',')
# y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

X = np.loadtxt('x_time_sample_F1000toT338.csv', delimiter=',')
y = np.loadtxt('y_time_label_F1000toT338.csv', delimiter=',', dtype='int8')

sc = StandardScaler()
X = sc.fit_transform(X)

# -------------------------------------------------
# prediction: k_fold
# -------------------------------------------------
# -------------------------
# select_model
# -------------------------

# model = SVC(kernel='rbf', C=6000, gamma=0.001, probability=True)
# model_name = 'SVM'

model = RandomForestClassifier(n_estimators=500, criterion='entropy')
model_name = 'RandomForest'

# model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
# model_name = 'Tree'

# model = KNeighborsClassifier(n_neighbors=3)
# model_name = '3NN'

# model = KNeighborsClassifier(n_neighbors=1)
# model_name = '1NN'
# -------------------------
test_size = 0.3
significance = 0.3
result_summary = []
# -------------------------------------------------------------------
# prediction with significance

# s_folder = StratifiedKFold(n_splits=10, shuffle=True)
# for index, (train, test) in enumerate(s_folder.split(X, y)):
#     X_train, X_test = X[train], X[test]
#     y_train, y_test = y[train], y[test]
#     x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(X_train, y_train, test_size=test_size, shuffle=True)
#     y_test = y_test.reshape((-1, 1))
#
#     lda = LinearDiscriminantAnalysis(n_components=9)
#     x_train_lda = lda.fit_transform(x_train_sp, y_train_sp)
#     x_cal_lda = lda.transform(x_cal)
#     x_test_lda = lda.transform(X_test)
#
#     nc = NcFactory.create_nc(model=model)
#     icp = IcpClassifier(nc)
#
#     icp.fit(x_train_lda, y_train_sp)
#     icp.calibrate(x_cal_lda, y_cal)
#     prediction = icp.predict(x_test_lda, significance=None)
#     print(prediction)
#
#     result = [1-class_mean_errors(prediction, y_test, significance=significance),
#               class_avg_c(prediction, y_test, significance=significance)]
#     if index == 0:
#         result_summary = result
#     else:
#         result_summary = np.vstack((result_summary, result))
#     print('\nICP')
#     print('Accuracy: {}'.format(result[0]))
#     print('Average count: {}'.format(result[1]))
#
# df_summary = pd.DataFrame(result_summary, columns=['Accuracy', 'Average_count'])
#
# summary_path = path + '/summary/icp/lda_' + str(test_size) + '/'
# summary_file = summary_path + 'icp_lda' + model_name + '.csv'
# if os.path.exists(summary_path) is not True:
#     os.makedirs(summary_path)
# if os.path.exists(summary_file) is True:
#     os.remove(summary_file)

#-----------------------------------------------------------
# force_prediction
sum_accuracy = []
time = 1
s_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
for index, (train, test) in enumerate(s_folder.split(X, y)):
    print(time)
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(X_train, y_train, test_size=test_size, shuffle=True)
    y_test = y_test.reshape((-1, 1))

    lda = LinearDiscriminantAnalysis(n_components=5)
    x_train_lda = lda.fit_transform(x_train_sp, y_train_sp)
    x_cal_lda = lda.transform(x_cal)
    x_test_lda = lda.transform(X_test)

    nc = NcFactory.create_nc(model=model)
    icp = IcpClassifier(nc)

    icp.fit(x_train_lda, y_train_sp)
    icp.calibrate(x_cal_lda, y_cal)
    prediction = icp.predict(x_test_lda, significance=None)
    force_value = 1-force_mean_errors(prediction, y_test)

    sum_accuracy.append(force_value)
    time += 1

print(model_name + ':')
print(np.mean(sum_accuracy))

#     result = [force_value]
#     if index == 0:
#         result_summary = result
#     else:
#         result_summary = np.vstack((result_summary, result))
#     print('\nICP_Force')
#     if np.unique(y_test).shape[0] == 10:
#         print('True')
#     else:
#         print('Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('Accuracy: {}'.format(result[0]))
#
# df_summary = pd.DataFrame(result_summary, columns=['Accuracy'])
#
# summary_path = path + '/summary/icp/lda_force/'
# summary_file = summary_path + 'icp_' + model_name + '.csv'
# if os.path.exists(summary_path) is not True:
#     os.makedirs(summary_path)
# if os.path.exists(summary_file) is True:
#     os.remove(summary_file)
#
# df_summary.to_csv(summary_file)
# print(df_summary)

