# -*- coding: utf-8 -*-
# @Time    : 2017/12/7 16:07
# @Author  : LeonHardt
# @File    : LDA.py


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plot_draw import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    X_train = np.loadtxt('x_sample.csv', delimiter=',')
    y_train = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    lda = LinearDiscriminantAnalysis(n_components=3)
    X_train_lda = lda.fit_transform(X_train_std, y_train)

    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)

    # plot_decision_regions(X_train_lda, y_train, classifier=lr)
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], X_train_lda[:, 2], marker='o', c=y_train)
    #
    # plt.xlabel('LD1')
    # plt.ylabel('LD2')
    # plt.legend(loc='lower left')
    plt.show()

