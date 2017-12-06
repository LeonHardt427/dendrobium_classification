#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/20 上午10:10
# @Author  : LeonHardt
# @File    : off-line_prediction.py

"""
Classify different categories of dendrobium using machine learning methods

Methods:
    1. Perceptron
    2. Logistic Regression
    3. Support Vector Machine (linear kernel)
    4. Support Vector Machine (rbf kernel)
    5. Decision Tree
    6. Random Forest
"""
import numpy as np
import pandas as pd
import os
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

NAME_SAMPLE = "F500toT159"
DIR_SIZE = "/size02/" + NAME_SAMPLE + ".csv"
TEST_NUMBER = 100

path = os.getcwd()
"""
X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
"""

X_train = np.loadtxt("x_sample_train_02" + NAME_SAMPLE + ".csv", delimiter=',')
y_train = np.loadtxt("y_label_train_02" + NAME_SAMPLE + ".csv", delimiter=',', dtype='int8')
X_test = np.loadtxt("x_sample_test_02" + NAME_SAMPLE + ".csv", delimiter=',')
y_test = np.loadtxt("y_label_test_02" + NAME_SAMPLE + ".csv", delimiter=',', dtype='int8')

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


"""
Method 1: Perceptron
"""
ppn_error_sample = []
ppn = Perceptron(n_iter=4000, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_prediction = ppn.predict(X_test_std)
for sample_number in range(TEST_NUMBER):
    if y_test[sample_number] != y_prediction[sample_number]:
        ppn_error_sample.append(1)
    else:
        ppn_error_sample.append(0)
misclassified_number = ((y_test != y_prediction).sum())
ppn_accuracy = accuracy_score(y_test, y_prediction)
print('Perceptron accuracy is ' + str(ppn_accuracy) + ' ,and ' +
      str(misclassified_number) + 'sample(s) is misclassified')



"""
Method 2: Logistic Regression
"""

lr_error_sample = []
lr = LogisticRegression(C=0.1, random_state=0)
lr.fit(X_train_std, y_train)
y_prediction = lr.predict(X_test_std)
for sample_number in range(TEST_NUMBER):
    if y_test[sample_number] != y_prediction[sample_number]:
        lr_error_sample.append(1)
    else:
        lr_error_sample.append(0)
misclassified_number = ((y_test != y_prediction).sum())
lr_accuracy = accuracy_score(y_test, y_prediction)
print('Logistic Regression is ' + str(lr_accuracy) + ' ,and ' +
      str(misclassified_number) + 'sample(s) is misclassified')


"""
Method 3: Support Vector machine (linear kernel)
"""
svm_linear_error_sample = []
svm = SVC(kernel='rbf', C=80.0, gamma=0.0005, random_state=0)
svm.fit(X_train_std, y_train)
y_prediction = svm.predict(X_test_std)
for sample_number in range(TEST_NUMBER):
    if y_test[sample_number] != y_prediction[sample_number]:
        svm_linear_error_sample.append(1)
    else:
        svm_linear_error_sample.append(0)
misclassified_number = ((y_test != y_prediction).sum())
svm_linear_accuracy = accuracy_score(y_test, y_prediction)
print('SVM_Linear accuracy is ' + str(svm_linear_accuracy) + ' ,and ' +
      str(misclassified_number) + 'sample(s) is misclassified')


"""
Method 4: Support Vector machine (rbf kernel)

svm_rbf_error_sample = []
svm = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=0)
svm.fit(X_train_std, y_train)
y_prediction = svm.predict(X_test_std)
for sample_number in range(144):
    if y_test[sample_number] != y_prediction[sample_number]:
        svm_rbf_error_sample.append(1)
    else:
        svm_rbf_error_sample.append(0)
misclassified_number = ((y_test != y_prediction).sum())
svm_rbf_accuracy = accuracy_score(y_test, y_prediction)
print('SVM_RBF accuracy is ' + str(svm_rbf_accuracy) + ' ,and ' +
      str(misclassified_number) + 'sample(s) is misclassified')
"""
"""
Method 4: Tree
"""
tree_error_sample = []
tree = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=0)
tree.fit(X_train_std, y_train)
y_prediction = tree.predict(X_test_std)
for sample_number in range(TEST_NUMBER):
    if y_test[sample_number] != y_prediction[sample_number]:
        tree_error_sample.append(1)
    else:
        tree_error_sample.append(0)
misclassified_number = ((y_test != y_prediction).sum())
tree_accuracy = accuracy_score(y_test, y_prediction)
print('Tree accuracy is ' + str(tree_accuracy) + ' ,and ' +
      str(misclassified_number) + 'sample(s) is misclassified')

"""
Method 5: Random Forest
"""
forest_error_sample = []
forest = RandomForestClassifier(criterion='entropy', n_estimators=6, random_state=1, n_jobs=2)
forest.fit(X_train_std, y_train)
y_prediction = forest.predict(X_test_std)
for sample_number in range(TEST_NUMBER):
    if y_test[sample_number] != y_prediction[sample_number]:
        forest_error_sample.append(1)
    else:
        forest_error_sample.append(0)
misclassified_number = ((y_test != y_prediction).sum())
forest_accuracy = accuracy_score(y_test, y_prediction)
print('Forest accuracy is ' + str(forest_accuracy) + ' ,and ' +
      str(misclassified_number) + 'sample(s) is misclassified')

acc_series = pd.Series([ppn_accuracy, lr_accuracy, svm_linear_accuracy, tree_accuracy, forest_accuracy],
                       index=['ppn', 'lr', 'svm', 'tree', 'forest'])

if os.path.exists(path + "/size02/" + NAME_SAMPLE) is not True:
    os.makedirs(path + "/size02/" + NAME_SAMPLE)

np.savetxt(path + "/size02/" + NAME_SAMPLE + '/ppn.txt', np.array(ppn_error_sample))
np.savetxt(path + "/size02/" + NAME_SAMPLE + '/lr.txt', np.array(lr_error_sample))
np.savetxt(path + "/size02/" + NAME_SAMPLE + '/svm.txt', np.array(svm_linear_error_sample))
# np.savetxt('./prediction_result/svm-rbf.txt', np.array(svm_rbf_error_sample))
np.savetxt(path + "/size02/" + NAME_SAMPLE + '/tree.txt', np.array(tree_error_sample))
np.savetxt(path + "/size02/" + NAME_SAMPLE + '/forest.txt', np.array(forest_error_sample))
acc_series.to_csv(path + DIR_SIZE)



