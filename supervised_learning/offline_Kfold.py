#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/23 下午9:23
# @Author  : LeonHardt
# @File    : offline_K_fold.py


"""
Classify different categories of dendrobium using machine learning methods

Methods:
    1. Perceptron
    2. Logistic Regression
    3. Support Vector Machine (linear kernel)
    4. Support Vector Machine (rbf kernel)
    5. Decision Tree
    6. Random Forest
    7. K-nearest neighbour
"""

import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X = np.loadtxt('x_time_sample_F1000toT338.csv', delimiter=',')
y = np.loadtxt('y_time_label_F1000toT338.csv', delimiter=',', dtype='int8')

K_fold = StratifiedKFold(n_splits=10, random_state=0)


# """
# Method 1: Perceptron
# """
# pipe_ppn = Pipeline([('scl', StandardScaler()), ('ppn', Perceptron(n_iter=100, eta0=0.1))])
# ppn_score = []
# fold_times = 1
# for train_index, test_index in K_fold.split(X, y):
#     pipe_ppn.fit(X[train_index], y[train_index])
#     score = pipe_ppn.score(X[test_index], y[test_index])
#     fold_times += 1
#     ppn_score.append(score)
#     # print('Perceptron: Fold: %s, Class dist.: %s, Acc: %.3f' % (fold_times, np.bincount(y[train_index]), score))
# ppn_score_cv = [np.mean(ppn_score), np.std(ppn_score)]
# print('Perceptron CV accuracy: %s +/- %s' % (ppn_score_cv[0], ppn_score_cv[1]))

# """
# Method 2: Logistic Regression
# """
# pipe_lr = Pipeline([('scl', StandardScaler()), ('lr', LogisticRegression(C=1000.0))])
# lr_score = []
# fold_times = 1
# for train_index, test_index in K_fold.split(X, y):
#     pipe_lr.fit(X[train_index], y[train_index])
#     score = pipe_lr.score(X[test_index], y[test_index])
#     fold_times += 1
#     lr_score.append(score)
#     # print('LOGISTIC: Fold: %s, Class dist.: %s, Acc: %.3f' % (fold_times, np.bincount(y[train_index]), score))
# lr_score_cv = [np.mean(lr_score), np.std(lr_score)]
# print('LOGISTIC CV accuracy: %s +/- %s' % (lr_score_cv[0], lr_score_cv[1]))

# """
# Method 3: Support Vector machine (linear kernel)
# """
# pipe_svm_linear = Pipeline([('scl', StandardScaler()), ('svm_linear', SVC(kernel='linear', C=1.0))])
# svm_linear_score = []
# fold_times = 1
# for train_index, test_index in K_fold.split(X, y):
#     pipe_svm_linear.fit(X[train_index], y[train_index])
#     score = pipe_svm_linear.score(X[test_index], y[test_index])
#     fold_times += 1
#     svm_linear_score.append(score)
#     # print('SVM_LINEAR: Fold: %s, Class dist.: %s, Acc: %.3f' % (fold_times, np.bincount(y[train_index]), score))
# svm_linear_score_cv = [np.mean(svm_linear_score), np.std(svm_linear_score)]
# print('SVM_LINEAR CV accuracy: %s +/- %s' % (svm_linear_score_cv[0], svm_linear_score_cv[1]))

"""
Method 4: Support Vector machine (rbf kernel)
"""
# pipe_svm_rbf = Pipeline([('scl', StandardScaler()), ('svm_rbf', SVC(kernel='rbf', C=60.0, gamma=0.001))])
# svm_rbf_score = []
# fold_times = 1
# for train_index, test_index in K_fold.split(X, y):
#     pipe_svm_rbf.fit(X[train_index], y[train_index])
#     score = pipe_svm_rbf.score(X[test_index], y[test_index])
#     fold_times += 1
#     svm_rbf_score.append(score)
#     # print('SVM_RBF: Fold: %s, Class dist.: %s, Acc: %.3f' % (fold_times, np.bincount(y[train_index]), score))
# svm_rbf_score_cv = [np.mean(svm_rbf_score), np.std(svm_rbf_score)]
# print('SVM_RBF CV accuracy: %s +/- %s' % (svm_rbf_score_cv[0], svm_rbf_score_cv[1]))

# """
# Method 5: Decision Tree
# """
# pipe_tree = Pipeline([('scl', StandardScaler()), ('tree', DecisionTreeClassifier(criterion='entropy', max_depth=8))])
# tree_score = []
# fold_times = 1
# for train_index, test_index in K_fold.split(X, y):
#     pipe_tree.fit(X[train_index], y[train_index])
#     score = pipe_tree.score(X[test_index], y[test_index])
#     fold_times += 1
#     tree_score.append(score)
#     # print('Tree: Fold: %s, Class dist.: %s, Acc: %.3f' % (fold_times, np.bincount(y[train_index]), score))
# tree_score_cv = [np.mean(tree_score), np.std(tree_score)]
# print('Tree CV accuracy: %s +/- %s' % (tree_score_cv[0], tree_score_cv[1]))

"""
Method 6: Random Forest
"""
pipe_forest = Pipeline([('scl', StandardScaler()), ('forest', RandomForestClassifier(criterion='entropy', max_depth=500))])
forest_score = []
fold_times = 1
for train_index, test_index in K_fold.split(X, y):
    pipe_forest.fit(X[train_index], y[train_index])
    score = pipe_forest.score(X[test_index], y[test_index])
    fold_times += 1
    forest_score.append(score)
    # print('Forest: Fold: %s, Class dist.: %s, Acc: %.3f' % (fold_times, np.bincount(y[train_index]), score))
forest_score_cv = [np.mean(forest_score), np.std(forest_score)]
print('Forest CV accuracy: %s +/- %s' % (forest_score_cv[0], forest_score_cv[1]))

# """
# Method 7: KNN
# """
# pipe_knn = Pipeline([('scl', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=1))])
# knn_score = []
# fold_times = 1
# for train_index, test_index in K_fold.split(X, y):
#     pipe_knn.fit(X[train_index], y[train_index])
#     score = pipe_knn.score(X[test_index], y[test_index])
#     fold_times += 1
#     knn_score.append(score)
# knn_score_cv = [np.mean(knn_score), np.std(knn_score)]
# print('KNN CV accuracy: %s +/- %s' % (knn_score_cv[0], knn_score_cv[1]))


# np.savetxt('ppn.txt', np.array(ppn_error_sample))
# np.savetxt('lr.txt', np.array(lr_error_sample))
# np.savetxt('svm-linear.txt', np.array(svm_linear_error_sample))
# np.savetxt('svm-rbf.txt', np.array(svm_rbf_error_sample))
# np.savetxt('tree.txt', np.array(tree_error_sample))
# np.savetxt('forest.txt', np.array(forest_error_sample))

