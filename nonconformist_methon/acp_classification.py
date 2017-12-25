# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 20:32
# @Author  : LeonHardt
# @File    : acp_classification.py


import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassifierNc
from nonconformist.icp import IcpClassifier
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier
from nonconformist.evaluation import class_mean_errors, class_avg_c

# -----------------------------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------------------------
# data = load_iris()

X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

sc = StandardScaler()

X = sc.fit_transform(X)

# idx = np.random.permutation(y.size)
# train = idx[:int(2 * idx.size / 3)]
# test = idx[int(2 * idx.size / 3):]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

truth = y_test.reshape(-1, 1)
columns = ['C-{}'.format(i) for i in np.unique(y)] + ['truth']
significance = 0.45
classification_method = DecisionTreeClassifier()

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

# -----------------------------------------------------------------------------
# Train, predict and evaluate
# -----------------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test, significance=None)
    table = np.hstack((prediction, truth))
    df = pd.DataFrame(table, columns=columns)
    print('\n{}'.format(name))
    print('Error rate: {}'.format(class_mean_errors(prediction, truth, significance)))
    print('class average count: {}'.format(class_avg_c(prediction, truth, significance)))
    # print(df)
