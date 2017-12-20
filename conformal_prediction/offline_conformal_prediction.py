# -*- coding: utf-8 -*-
# @Time    : 2017/12/15 16:43
# @Author  : LeonHardt
# @File    : offline_conformal_prediction.py


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold
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

    X = np.loadtxt('x_sample.csv', delimiter=',')
    y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

    sc = StandardScaler()
    sc.fit(X)

    """
    Using train_test_split to train the model
    """
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #

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

    # prediction = np.loadtxt('1NN_prediction.txt', delimiter=',', dtype='int8')
    # label = np.loadtxt('1NN_y_train.txt', delimiter=',', dtype='int8')

    """
    Using StratifiedKFold
    """
    for k_num in range(1, 4, 1):

        s_folder = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
        accuracy = []

        split_time = 1
        for train_index, test_index in s_folder.split(X, y):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_train_std = sc.transform(x_train)   # standardization
            x_test_std = sc.transform(x_test)

            lda = LinearDiscriminantAnalysis(n_components=9)        # LDA
            lda.fit(x_train_std, y_train)
            x_train_std_lda = lda.transform(x_train_std)
            x_test_std_lda = lda.transform(x_test_std)

            print('The '+str(split_time)+' time')          # count
            for i in range(4):
                print(' ')
            split_time = split_time + 1

            p_value = ConformalPredictionKnn(x_train_std_lda, x_test_std_lda, y_train, k=k_num)     #prediction
            knn_prediction = CP_KNN_offline(p_value)

            score = score_prediction(knn_prediction, y_test)        # score
            accuracy.append(score)

        np.savetxt('acc_k'+str(k_num)+'_lda.txt', accuracy, delimiter=',')








