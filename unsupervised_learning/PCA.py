# -*- coding: utf-8 -*-
# @Time    : 2017/12/7 9:47
# @Author  : LeonHardt
# @File    : PCA.py

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from plot_draw import plot_decision_regions
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    X_train = np.loadtxt('x_sample.csv', delimiter=',')
    y_train = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    # X_test

    lr = LogisticRegression()
    lr.fit(X_train_pca, y_train)

    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(loc='lower left')
    plt.show()
