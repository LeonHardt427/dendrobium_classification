#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/28 下午2:19
# @Author  : LeonHardt
# @File    : parameter_tuning.py


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

"""
choose the classification method:
Methods:
    1. Perceptron       ------------------------         'ppn'
    2. Logistic Regression -------------------------     'lr'
    3. Support Vector Machine (linear kernel)----------  'svm'
    4. Support Vector Machine (rbf kernel)-----------    'svm
    5. Decision Tree        -------------------          'tree'
    6. Random Forest    ------------------------         'forest'
"""
# Set the parameter of the classification method
METHOD = 'FOREST'
REPORT_NEED = 0
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
max_depth_range = [4, 8, 12, 16, 20]

"""
if method is 'LR':
"""
if METHOD is 'LR':
    pip_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(random_state=1))])

    param_grid = [{'clf__C': param_range}]
    gs = GridSearchCV(estimator=pip_lr, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
    gs = gs.fit(X, y)

# Show the full report (Yes: 1 ; No: 0)
    if REPORT_NEED is 1:
        print()
        for number in range(len(param_range)):
            print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][number],
                                                gs.cv_results_['std_test_score'][number],
                                                gs.cv_results_['params'][number],))
    print("Best parameters of Logistic Regression is:")
    print(gs.best_params_)
    print(gs.best_score_)

"""
if method is 'SVM':
"""
if METHOD is 'SVM':
    pip_svm = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
    param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                  {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
    gs = GridSearchCV(estimator=pip_svm, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
    gs = gs.fit(X, y)

# Show the full report (Yes: 1 ; No: 0)
#     if REPORT_NEED is 1:
#         print()
#         for number in range(len(param_range)):
#             print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][number],
#                                                 gs.cv_results_['std_test_score'][number],
#                                                 gs.cv_results_['params'][number],))
    print("Best parameters of SVM is:")
    print(gs.best_params_)
    print(gs.best_score_)


"""
if method is 'tree':
"""
if METHOD is 'TREE':
    pip_tree = Pipeline([('scl', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=1))])
    param_grid = [{'clf__criterion': ['entropy'], 'clf__max_depth': max_depth_range}]
    gs = GridSearchCV(estimator=pip_tree, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
    gs = gs.fit(X, y)

# Show the full report (Yes: 1 ; No: 0)
    if REPORT_NEED is 1:
        print()
        for number in range(len(param_range)):
            print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][number],
                                                gs.cv_results_['std_test_score'][number],
                                                gs.cv_results_['params'][number],))
    print("Best parameters of Tree is:")
    print(gs.best_params_)
    print(gs.best_score_)

"""
if method is 'forest':
"""
if METHOD is 'FOREST':
    pip_forest = Pipeline([('scl', StandardScaler()), ('clf', RandomForestClassifier(random_state=1))])
    param_grid = [{'clf__criterion': ['entropy'], 'clf__max_depth': max_depth_range}]
    gs = GridSearchCV(estimator=pip_forest, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
    gs = gs.fit(X, y)

# Show the full report (Yes: 1 ; No: 0)
    if REPORT_NEED is 1:
        print()
        for number in range(len(param_range)):
            print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][number],
                                                gs.cv_results_['std_test_score'][number],
                                                gs.cv_results_['params'][number],))
    print("Best parameters of FOREST is:")
    print(gs.best_params_)
    print(gs.best_score_)




