# -*- coding: utf-8 -*-
# @Time    : 2017/12/21 10:30
# @Author  : LeonHardt
# @File    : lda_visualiztion.py


import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

def lda_visualization_2D(feature_set, label_set):
    """
    draw plot using LDA_2D (n_components=2)

    :param feature_set: features of the sample
    :param label_set:   label information
    :param component:  the n_components of LinearDiscriminantAnalysis
    :return: None
    """
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'aqua', 'orange', 'pink', 'snow', 'greenyellow')


    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(feature_set, label_set)
    x = lda.transform(feature_set)

    plt.scatter(x[:, 0], x[:, 1], marker='o', c=label_set)
    plt.show()

def lda_visualization_3D(feature_set, label_set):
    """
    draw plot using LDA_3D (n_components=2)

    :param feature_set: features of the sample
    :param label_set:   label information
    :param component:  the n_components of LinearDiscriminantAnalysis
    :return: None
    """
    lda = LinearDiscriminantAnalysis(n_components=3)
    lda.fit(feature_set, label_set)
    x = lda.transform(feature_set)

    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o', c=label_set)
    plt.show()