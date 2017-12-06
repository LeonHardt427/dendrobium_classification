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


# X = np.loadtxt('x_sample.csv', delimiter=',')
# y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

"""
Draw_plot function: Draw a plot about classification
"""


def plot_decision_regions(sample, label, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(label))])

    # plot the decision surface
    x1_min, x1_max = sample[:, 0].min() - 1, sample[:, 0].max() + 1
    x2_min, x2_max = sample[:, 0].min() - 1, sample[:, 0].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           (np.arange(x2_min, x2_max, resolution)))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all the samples
#    x_test_set, y_test_set = sample[test_idx, :], labels[test_idx]
    for idx, cl in enumerate(np.unique(label)):
        plt.scatter(x=sample[label == cl, 0], y=sample[label == cl, 1],
                    alpha=0.6, c=cmap(idx), edgecolor='black',
                    marker=markers[idx], label=cl)

    # highlight test sample
    if test_idx:
        x_test, y_test = sample[test_idx, :], label[test_idx]
        plt.scatter(x_test[:, 0], x_test[:, 1], c='',
                    alpha=1.0, edgecolor='black', linewidths=1, marker='o',
                    s=55, label='test_set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

"""
Method 1: Perceptron
"""

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_prediction = ppn.predict(X_test_std)
misclassified_number = ((y_test != y_prediction).sum())
ppn_accuracy = accuracy_score(y_test, y_prediction)
print('Perceptron accuracy is ' + str(ppn_accuracy) + ' ,and ' +
      str(misclassified_number) + 'sample(s) is misclassified')

plot_decision_regions(sample=X_combined_std, label=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.title('Perceptron')                     # Title ------Perceptron
plt.xlabel('petal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

"""
Method 2: Logistic Regression
"""

lr = LogisticRegression(C=1000, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(sample=X_combined_std, label=y_combined, classifier=lr, test_idx=range(105, 150))
plt.title('Logistic Regression')           # Title ------- Logistic Regression
plt.xlabel('petal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

"""
Method 3: Support Vector machine (linear kernel)
"""
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(sample=X_combined_std, label=y_combined, classifier=svm, test_idx=range(105, 150))
plt.title('Support Vector Machine (linear kernel)')           # Title ------- Support Vector Machine (linear kernel)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

"""
Method 4: Support Vector machine (rbf kernel)
"""
svm = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(sample=X_combined_std, label=y_combined, classifier=svm, test_idx=range(105, 150))
plt.title('Support Vector Machine (RBF kernel)')           # Title ------- Support Vector Machine (rbf kernel)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

"""
Method 4: Support Vector machine (rbf kernel)
"""
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train_std, y_train)
plot_decision_regions(sample=X_combined_std, label=y_combined, classifier=tree, test_idx=range(105, 150))
plt.title('Decision Tree')           # Title ------- Decision Tree
plt.xlabel('petal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

"""
Method 5: Random Forest
"""
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train_std, y_train)
plot_decision_regions(sample=X_combined_std, label=y_combined, classifier=forest, test_idx=range(105, 150))
plt.title('Random Forest')           # Title ------- Random Forest
plt.xlabel('petal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

print(X, X.shape)
print(y, y.shape)
