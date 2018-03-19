# -*- coding: utf-8 -*-
# @Time    : 2018/3/18 21:10
# @Author  : LeonHardt
# @File    : prediction_pvalue.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory, ClassifierNc
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier

from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.acp import BootstrapConformalClassifier
from force_value import force_mean_errors

# ----------------------------------------
# preprocessing
# -----------------------------------------
path = os.getcwd()

X = np.loadtxt('x_time_sample_F1000toT338.csv', delimiter=',')
y = np.loadtxt('y_time_label_F1000toT338.csv', delimiter=',')
# X = np.loadtxt('x_sample.csv', delimiter=',')
# y = np.loadtxt('y_label.csv', delimiter=',')

sc = StandardScaler()
X = sc.fit_transform(X)

# --------------------------------------------
# prediction
# --------------------------------------------
simple_model = RandomForestClassifier(n_estimators=500, criterion='entropy')
model_name = "RF(500)"

# simple_model = SVC(C=60.0, gamma=0.001, probability=True)
# model_name = "SVM(60,0.001)"
# -----------------------------------------------------------------------------
# Define models
# -----------------------------------------------------------------------------

models = {  'ACP-RandomSubSampler'  : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model))),
                                        RandomSubSampler()),
            'ACP-CrossSampler'      : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model))),
                                        CrossSampler()),
            'ACP-BootstrapSampler'  : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model))),
                                        BootstrapSampler()),
            'CCP'                   : CrossConformalClassifier(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model)))),
            'BCP'                   : BootstrapConformalClassifier(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model)))),
          }

# -------------------------------------------------------------------------------
# sig_error_lda
# ----------------------------------------------------------------------------
# s_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# for framework_name, model in models.items():
#     print(framework_name + ' is starting:')
#     for num, (train, test) in enumerate(s_folder.split(X, y)):
#         x_train, x_test = X[train], X[test]
#         y_train, y_test = y[train], y[test]
#         truth = y_test.reshape((-1, 1))
#
#         lda = LinearDiscriminantAnalysis(n_components=9)
#         x_train_lda = lda.fit_transform(x_train, y_train)
#         x_test_lda = lda.transform(x_test)
#
#         model.fit(x_train_lda, y_train)
#         prediction = model.predict(x_test_lda, significance=None)
#
#         for sig in np.arange(0, 1.0001, 0.005):
#             print(framework_name + ': sig = ' + str(sig))

        # ------------------------------------------
        # ICP
        # x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(x_train, y_train, test_size=0.3, shuffle=True,
        #                                                         random_state=1)
        # nc = NcFactory.create_nc(model=simple_model)
        # conformal_model = IcpClassifier(nc)
        # conformal_model.fit(x_train_sp, y_train_sp)
        # conformal_model.calibrate(x_cal, y_cal)
        # table = np.hstack((prediction, truth))

        #     result = [sig, class_mean_errors(prediction, truth, significance=sig),
        #               class_avg_c(prediction, truth, significance=sig)]
        #     if sig == 0:
        #         summary = result
        #     else:
        #         summary = np.vstack((summary, result))
        #
        # df_summary = pd.DataFrame(summary, columns=['sig', 'Accuracy', 'Average_count'])
        #
        # save_path = os.getcwd() + '/summary/time/' + model_name + '/' + framework_name + '/'
        # if os.path.exists(save_path) is not True:
        #     os.makedirs(save_path)
        # save_file = save_path + framework_name + '_' + str(num) + '.csv'
        # if os.path.exists(save_file):
        #     os.remove(save_file)
        # df_summary.to_csv(save_file)

    # print(df_summary)
    # print(df_summary['Accuracy'].mean())
    # print(type(df_summary['Accuracy'].mean()))
#
# summary_path = path + '/summary/bcp/'
# summary_file = summary_path + 'bcp_' + model_name + '.csv'
# if os.path.exists(summary_path) is not True:
#     os.makedirs(summary_path)
# if os.path.exists(summary_file) is True:
#     os.remove(summary_file)

# -------------------------------------------------------------------------------
# sig_error_
# ----------------------------------------------------------------------------
prediction_summary = []

s_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
for framework_name, model in models.items():
    print(framework_name + ' is starting:')
# for i in range(1):
    for num, (train, test) in enumerate(s_folder.split(X, y)):
        print('the ' + str(num) + ' time is starting')
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        truth = y_test.reshape((-1, 1))

        # ---------------------------------------------------
        # ACP
        # --------------------------------------------------
        model.fit(x_train, y_train)
        prediction = model.predict(x_test, significance=None)
        table = np.hstack((prediction, truth))

        # ------------------------------------------
        # ICP
        # -------------------------------------------
        # x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(x_train, y_train, test_size=0.3, shuffle=True,
        #                                                         random_state=1)
        # nc = NcFactory.create_nc(model=simple_model)
        # conformal_model = IcpClassifier(nc)
        # conformal_model.fit(x_train_sp, y_train_sp)
        # conformal_model.calibrate(x_cal, y_cal)
        # prediction = conformal_model.predict(x_test, significance=None)
        # table = np.hstack((prediction, truth))

        # ---------------------------------------------------------------------------
        if num == 0:
            prediction_summary = table
        else:
            prediction_summary = np.vstack((prediction_summary, table))

    save_path = os.getcwd() + '/prediction/ACP_' + model_name + '/'
    if os.path.exists(save_path) is not True:
        os.makedirs(save_path)
    np.savetxt(save_path + '/' + framework_name + '.txt', prediction_summary, delimiter=',')

