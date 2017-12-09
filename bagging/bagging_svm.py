#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/28 下午8:02
# @Author  : LeonHardt
# @File    : Bagging_tree.py

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from draw import draw_line
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    X = np.loadtxt('x_sample.csv', delimiter=',')
    y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')
    le = LabelEncoder()
    y = le.fit_transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    svm = SVC(kernel='linear', C=1.0, gamma=0.2)
    scores_simple = cross_val_score(estimator=svm, X=X, y=y, cv=10, n_jobs=1)
    print(np.mean(scores_simple))

    # result = []
    # for time in range(1, 500, 5):
    #     print('the %d times' % time)
    #     bag = BaggingClassifier(base_estimator=svm, n_estimators=time, max_samples=1.0, max_features=1.0, bootstrap=True,
    #                         bootstrap_features=False, n_jobs=1, random_state=1)
    #     scores = cross_val_score(estimator=bag, X=X, y=y, cv=10, n_jobs=-1)
    #     print(np.mean(scores))
    #     result.append([time, np.mean(scores), np.std(scores)])
    #
    # np.savetxt('result_svm.txt', result, delimiter=',')
    # print('done')
    #
    # '''draw picture'''
    # draw_line(result[:][1], result[:][2])

    # print('CV scores is %s' % scores)
    # print('CV accuracy is %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    # tree.fit(X_train, y_train)
    # y_train_pred = tree.predict(X_train)
    # y_test_pred = tree.predict(X_test)
    # tree_train = accuracy_score(y_train, y_train_pred)
    # tree_test = accuracy_score(y_test, y_test_pred)
    # print('DecisionTree train/test accuracy is %.3f/%.3f' % (tree_train, tree_test))

    # bag.fit(X_train, y_train)
    # y_train_bag = bag.predict(X_train)
    # y_test_bag = bag.predict(X_test)
    # bag_train = accuracy_score(y_train, y_train_bag)
    # bag_test = accuracy_score(y_test, y_test_bag)
    # print('Bagging train/test accuaracy is %.3f/%.3f' % (bag_train, bag_test))


