#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/28 下午8:02
# @Author  : LeonHardt
# @File    : Bagging_tree.py

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':

    X = np.loadtxt('x_sample.csv', delimiter=',')
    y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')
    le = LabelEncoder()
    y = le.fit_transform(y)

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=1)

    result = []
    for time in range(1, 500, 5):
        print('the %d times' % time)
        bag = BaggingClassifier(base_estimator=tree, n_estimators=time, max_samples=0.8, max_features=0.5, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
        scores = cross_val_score(estimator=bag, X=X, y=y, cv=10, n_jobs=-1)
        result.append([time, np.mean(scores), np.std(scores)])

    np.savetxt('result_tree', result)
    print('done')


