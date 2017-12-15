# -*- coding: utf-8 -*-
# @Time    : 2017/12/15 16:43
# @Author  : LeonHardt
# @File    : offline_conformal_prediction.py

import numpy as np
import matplotlib.pyplot as plt
from cp_knn import ConformalPredictionKnn
if __name__ == '__main__':

    X = np.loadtxt('x_sample.csv', delimiter=',')
    y = np.loadtxt('y_label.csv',delimiter=',')



