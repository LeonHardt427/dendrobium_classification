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
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

# iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
# y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


"""
Method 1: Perceptron
"""
ppn_error_sample = []
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_prediction = ppn.predict(X_test_std)
for sample_number in range(144):
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
lr = LogisticRegression(C=1000, random_state=0)
lr.fit(X_train_std, y_train)
y_prediction = lr.predict(X_test_std)
for sample_number in range(144):
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
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
y_prediction = svm.predict(X_test_std)
for sample_number in range(144):
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
"""
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
Method 4: Support Vector machine (rbf kernel)
"""
tree_error_sample = []
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train_std, y_train)
y_prediction = tree.predict(X_test_std)
for sample_number in range(144):
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
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train_std, y_train)
y_prediction = forest.predict(X_test_std)
for sample_number in range(144):
    if y_test[sample_number] != y_prediction[sample_number]:
        forest_error_sample.append(1)
    else:
        forest_error_sample.append(0)
misclassified_number = ((y_test != y_prediction).sum())
forest_accuracy = accuracy_score(y_test, y_prediction)
print('Forest accuracy is ' + str(forest_accuracy) + ' ,and ' +
      str(misclassified_number) + 'sample(s) is misclassified')


np.savetxt('ppn.txt', np.array(ppn_error_sample))
np.savetxt('lr.txt', np.array(lr_error_sample))
np.savetxt('svm-linear.txt', np.array(svm_linear_error_sample))
np.savetxt('svm-rbf.txt', np.array(svm_rbf_error_sample))
np.savetxt('tree.txt', np.array(tree_error_sample))
np.savetxt('forest.txt', np.array(forest_error_sample))



print(X, X.shape)
print(y, y.shape)
