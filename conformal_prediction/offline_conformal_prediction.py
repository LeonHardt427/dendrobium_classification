# -*- coding: utf-8 -*-
# @Time    : 2017/12/15 16:43
# @Author  : LeonHardt
# @File    : offline_conformal_prediction.py


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cp_knn import ConformalPredictionKnn, CP_KNN_offline


def score_prediction(p_value_predict, y_test):
    p_value_predict = p_value_predict.ravel()
    y_test = y_test.ravel()
    test_sample_number = p_value_predict.shape[0]
    right_number = 0
    for num in range(test_sample_number):
        if p_value_predict[num] == y_test[num]:
            right_number = right_number + 1

    score = right_number / test_sample_number
    return score


if __name__ == '__main__':

    # X = np.loadtxt('x_sample.csv', delimiter=',')
    # y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')
    # y = y.reshape((-1, 1))
    #
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #
    # sc = StandardScaler()
    # sc.fit(X)
    # x_train_std = sc.transform(x_train)
    # x_test_std = sc.transform(x_test)
    #
    # knn_value = ConformalPredictionKnn(x_train=x_train_std, x_test=x_test_std, y_train=y_train, k=1)
    # knn_prediction = CP_KNN_offline(knn_value)
    #
    # np.savetxt('1NN p_value.txt', knn_value, delimiter=',')
    # np.savetxt('1NN_y_train.txt', y_train, delimiter=',')
    # np.savetxt('1NN_prediction.txt', knn_prediction, delimiter=',')\
    #
    # score = score_prediction(knn_prediction, y_test)
    # print(score)
    prediction = np.loadtxt('1NN_prediction.txt', delimiter=',', dtype='int8')
    label = np.loadtxt('1NN_y_train.txt', delimiter=',', dtype='int8')

    score = score_prediction(prediction, label)
    print(score)








