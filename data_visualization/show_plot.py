# -*- coding: utf-8 -*-
# @Time    : 2017/12/21 11:10
# @Author  : LeonHardt
# @File    : show_plot.py

import numpy as np
from lda_visualization import lda_visualization_3D

if __name__ == '__main__':
    X = np.loadtxt('x_sample.csv', delimiter=',')
    y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

    lda_visualization_3D(X, y)

