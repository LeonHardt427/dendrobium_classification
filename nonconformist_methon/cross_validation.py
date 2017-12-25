# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 10:05
# @Author  : LeonHardt
# @File    : cross_validation.py


import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import  StandardScaler

from nonconformist.base import ClassifierAdapter
from nonconformist.icp import IcpClassifier
from nonconformist.nc import MarginErrFunc
from nonconformist.nc import ClassifierNc


from nonconformist.evaluation import cross_val_score
from nonconformist.evaluation import ClassIcpCvHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------------
    X = np.loadtxt('x_sample.csv', delimiter=',')
    y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

    sc = StandardScaler()
    X = sc.fit_transform(X)
    # data = load_iris()

    icp = IcpClassifier(ClassifierNc(ClassifierAdapter(RandomForestClassifier(n_estimators=100)),
                                     MarginErrFunc()))
    icp_cv = ClassIcpCvHelper(icp)

    scores = cross_val_score(icp_cv,
                             X,
                             y,
                             iterations=5,
                             folds=5,
                             scoring_funcs=[class_mean_errors, class_avg_c],
                             #significance_levels=[0.05, 0.1, 0.2]
                             )

    print('Classification: dendrobium')
    scores = scores.drop(['fold', 'iter'], axis=1)
    print(scores.groupby(['significance']).mean())


