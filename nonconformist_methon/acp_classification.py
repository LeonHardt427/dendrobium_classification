# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 20:32
# @Author  : LeonHardt
# @File    : acp_classification.py

import os
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , StratifiedKFold

from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassifierNc
from nonconformist.icp import IcpClassifier
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier
from force_value import force_mean_errors
from nonconformist.evaluation import class_mean_errors, class_avg_c, class_empty

# -----------------------------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------------------------
# data = load_iris()
path = os.getcwd()
X = np.loadtxt('x_time_sample_F1000toT338.csv', delimiter=',')
y = np.loadtxt('y_time_label_F1000toT338.csv', delimiter=',', dtype='int8')

sc = StandardScaler()

X = sc.fit_transform(X)

columns = ['C-{}'.format(i) for i in np.unique(y)] + ['truth']
significance = 0.45
# classification_method = SVC(C=60.0, gamma=0.001, kernel='rbf', probability=True)
classification_method = RandomForestClassifier(n_estimators=500, criterion='entropy')
file_name = 'dendrobium_RF(500).xls'

# classification_method = RandomForestClassifier(n_estimators=500, criterion='entropy')


ACP_Random = []
ACP_Cross = []
ACP_Boot = []
CCP = []
BCP = []
# -----------------------------------------------------------------------------
# Define models
# -----------------------------------------------------------------------------

models = {  'ACP-RandomSubSampler'  : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(classification_method))),
                                        RandomSubSampler()),
            'ACP-CrossSampler'      : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(classification_method))),
                                        CrossSampler()),
            'ACP-BootstrapSampler'  : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(classification_method))),
                                        BootstrapSampler()),
            'CCP'                   : CrossConformalClassifier(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(classification_method)))),
            'BCP'                   : BootstrapConformalClassifier(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(classification_method)))),
          }

# ------------------------------------------------------------------------------------------------------------------
# Train_test_split method:
# ------------------------------------------------------------------------------------------------------------------
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# truth = y_test.reshape(-1, 1)
# columns = ['C-{}'.format(i) for i in np.unique(y)] + ['truth']
# significance = 0.45
# classification_method = DecisionTreeClassifier()
#
#
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     prediction = model.predict(X_test, significance=None)
#     table = np.hstack((prediction, truth))
#     df = pd.DataFrame(table, columns=columns)
#     print('\n{}'.format(name))
#     print('Error rate: {}'.format(class_mean_errors(prediction, truth, significance)))
#     print('class average count: {}'.format(class_avg_c(prediction, truth, significance)))
#     # print(df)
#


# ------------------------------------------------------------------------------------------------------------------
# StratifiedKFold method
# ------------------------------------------------------------------------------------------------------------------

s_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
summary = []
summary_acc = []
time = 1
for train_index, test_index in s_folder.split(X, y):
    print('the ' + str(time) + 'time:')
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    truth = y_test.reshape(-1, 1)

    # lda = LinearDiscriminantAnalysis(n_components=9)
    # x_train_lda = lda.fit_transform(x_train, y_train)
    # x_test_lda = lda.transform(x_test)
    result = []
    for name, model in models.items():
        print(name + ' is predicting')
        model.fit(x_train, y_train)
        prediction = model.predict(x_test, significance=None)
        table = np.hstack((prediction, truth))
        accuracy = 1 - force_mean_errors(prediction, y_test)

        # df = pd.DataFrame(table, columns=columns)
        # print('\n{}'.format(name))
        # print('Error rate: {}'.format(class_mean_errors(prediction, truth, significance)))
        # print('class average count: {}'.format(class_avg_c(prediction, truth, significance)))
        # result = [name,
        #           class_mean_errors(prediction, truth, significance),
        #           class_avg_c(prediction, truth, significance),
        #           class_empty(prediction, truth, significance)]

        # if len(summary) == 0:
        #     summary = result
        # else:
        #     summary = np.vstack((summary, result))

        summary_acc.append(accuracy)
        if name == 'ACP-RandomSubSampler':
                ACP_Random.append(accuracy)
        elif name == 'ACP-CrossSampler':
                ACP_Cross.append(accuracy)
        elif name == 'ACP-BootstrapSampler':
                ACP_Boot.append(accuracy)
        elif name == 'CCP':
                CCP.append(accuracy)
        elif name == 'BCP':
                BCP.append(accuracy)
    time += 1


print('ACP_Random: ' + str(np.mean(ACP_Random)))
print('ACP-Cross: ' + str(np.mean(ACP_Cross)))
print('ACP-Boot: ' + str(np.mean(ACP_Boot)))
print('CCP: ' + str(np.mean(CCP)))
print('BCP: ' + str(np.mean(BCP)))
save_path = path + '/time/acp_svm/'
if os.path.exists(save_path) is not True:
    os.makedirs(save_path)
np.savetxt(save_path+'/acp-ran.txt', ACP_Random, delimiter=',')
np.savetxt(save_path+'/acp-cro.txt', ACP_Cross, delimiter=',')
np.savetxt(save_path+'/acp-boot.txt', ACP_Boot, delimiter=',')
np.savetxt(save_path+'/ccp.txt', CCP, delimiter=',')
np.savetxt(save_path+'/bcp.txt', BCP, delimiter=',')



# ------------------------------------------------------------------------------------------------------------------
# StratifiedKFold method_lda
# ------------------------------------------------------------------------------------------------------------------
#
# s_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# summary = []
# summary_acc = []
# time = 1
# for train_index, test_index in s_folder.split(X, y):
#     print('the ' + str(time) + 'time:')
#     x_train, x_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     truth = y_test.reshape(-1, 1)
#
#     lda = LinearDiscriminantAnalysis(n_components=9)
#     x_train_lda = lda.fit_transform(x_train, y_train)
#     x_test_lda = lda.transform(x_test)
#     result = []
#     for name, model in models.items():
#         print(name + ' is predicting')
#         model.fit(x_train_lda, y_train)
#         prediction = model.predict(x_test_lda, significance=None)
#         table = np.hstack((prediction, truth))
#         accuracy = 1 - force_mean_errors(prediction, y_test)
#
#         # df = pd.DataFrame(table, columns=columns)
#         # print('\n{}'.format(name))
#         # print('Error rate: {}'.format(class_mean_errors(prediction, truth, significance)))
#         # print('class average count: {}'.format(class_avg_c(prediction, truth, significance)))
#         # result = [name,
#         #           class_mean_errors(prediction, truth, significance),
#         #           class_avg_c(prediction, truth, significance),
#         #           class_empty(prediction, truth, significance)]
#
#         # if len(summary) == 0:
#         #     summary = result
#         # else:
#         #     summary = np.vstack((summary, result))
#
#         summary_acc.append(accuracy)
#         if name == 'ACP-RandomSubSampler':
#                 ACP_Random.append(accuracy)
#         elif name == 'ACP-CrossSampler':
#                 ACP_Cross.append(accuracy)
#         elif name == 'ACP-BootstrapSampler':
#                 ACP_Boot.append(accuracy)
#         elif name == 'CCP':
#                 CCP.append(accuracy)
#         elif name == 'BCP':
#                 BCP.append(accuracy)
#     time += 1
#
#
# print('ACP_Random: ' + str(np.mean(ACP_Random)))
# print('ACP-Cross: ' + str(np.mean(ACP_Cross)))
# print('ACP-Boot: ' + str(np.mean(ACP_Boot)))
# print('CCP: ' + str(np.mean(CCP)))
# print('BCP: ' + str(np.mean(BCP)))
# save_path = path + '/acp_RF/'
# if os.path.exists(save_path) is not True:
#     os.makedirs(save_path)
# np.savetxt(save_path+'/acp-ran.txt', ACP_Random, delimiter=',')
# np.savetxt(save_path+'/acp-cro.txt', ACP_Cross, delimiter=',')
# np.savetxt(save_path+'/acp-boot.txt', ACP_Boot, delimiter=',')
# np.savetxt(save_path+'/ccp.txt', CCP, delimiter=',')
# np.savetxt(save_path+'/bcp.txt', BCP, delimiter=',')
#



        # if name == 'ACP-RandomSubSampler':
        #     if index == 0:
        #         ACP_Random = result
        #     else:
        #         ACP_Random = np. vstack((ACP_Random, result))
        # elif name == 'ACP-CrossSampler':
        #     if index == 0:
        #         ACP_Cross = result
        #     else:
        #         ACP_Cross = np. vstack((ACP_Cross, result))
        # elif name == 'ACP-BootstrapSampler':
        #     if index == 0:
        #         ACP_Boot = result
        #     else:
        #         ACP_Boot = np. vstack((ACP_Boot, result))
        # elif name == 'CCP':
        #     if index == 0:
        #         CCP = result
        #     else:
        #         CCP = np.vstack((CCP, result))
        # elif name == 'BCP':
        #     if index == 0:
        #         BCP = result
        #     else:
        #         BCP = np.vstack((BCP, result))

# df_result = pd.DataFrame(summary, columns=['name', 'class_mean_errors', 'class_avg_c', 'class_empty'])
# df_result.to_excel(file_name, sheet_name='Sheet1')






