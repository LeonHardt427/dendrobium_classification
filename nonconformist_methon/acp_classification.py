# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 20:32
# @Author  : LeonHardt
# @File    : acp_classification.py


import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
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
from nonconformist.evaluation import class_mean_errors, class_avg_c, class_empty

# -----------------------------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------------------------
# data = load_iris()

X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

sc = StandardScaler()

X = sc.fit_transform(X)

columns = ['C-{}'.format(i) for i in np.unique(y)] + ['truth']
significance = 0.45
classification_method = DecisionTreeClassifier()
file_name = 'decision_tree.xls'

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
                                                ClassifierAdapter(classification_method))))
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

s_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
summary = []
for train_index, test_index in s_folder.split(X, y):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    truth = y_test.reshape(-1, 1)

    for name, model in models.items():
        result = []
        model.fit(x_train, y_train)
        prediction = model.predict(x_test, significance=None)
        table = np.hstack((prediction, truth))
        df = pd.DataFrame(table, columns=columns)
        print('\n{}'.format(name))
        print('Error rate: {}'.format(class_mean_errors(prediction, truth, significance)))
        print('class average count: {}'.format(class_avg_c(prediction, truth, significance)))
        result = [name,
                  class_mean_errors(prediction, truth, significance),
                  class_avg_c(prediction, truth, significance),
                  class_empty(prediction, truth, significance)]

        if len(summary) == 0:
            summary = result
        else:
            summary = np.vstack((summary, result))


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

df_result = pd.DataFrame(summary, columns=['name', 'class_mean_errors', 'class_avg_c', 'class_empty'])
df_result.to_excel(file_name, sheet_name='Sheet1')






