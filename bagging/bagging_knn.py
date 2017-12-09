#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/28 下午8:02
# @Author  : LeonHardt
# @File    : Bagging_knn.py

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    X = np.loadtxt('x_sample.csv', delimiter=',')
    y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')
    le = LabelEncoder()
    y = le.fit_transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')

    result = []
    pic = plt.figure(figsize=(8, 6), dpi=80)
    plt.ion()
    for time in range(1, 150, 2):
        plt.cla()
        print('the %d times' % time)
        bag = BaggingClassifier(base_estimator=knn, n_estimators=time, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
        scores = cross_val_score(estimator=bag, X=X, y=y, cv=10, n_jobs=-1)

        result.append([time, np.mean(scores), np.std(scores)])
        if time > 2:
            result_plot = np.array(result)
            x_data = result_plot[:, 0]
            y_data = result_plot[:, 1]

            plt.title('Bagging_2NN')
            plt.grid(True)
            plt.xlabel('Accuracy')
            plt.xlim(0, 170)
            plt.ylabel('Times')
            plt.ylim(0, 1)

            plt.plot(x_data, y_data, "b--", label='cos')
            plt.legend(loc="upper left", shadow=True)
            plt.pause(0.001)
    # result_plot = np.array(result)
    # x_data = result_plot[:, 0]
    # y_data = result_plot[:, 1]
    #
    #
    # plt.title('Bagging_2NN')
    # plt.grid(True)
    # plt.xlabel('Accuracy')
    # plt.xlim(0, 100)
    # plt.ylabel('Times')
    # plt.ylim(0, 1)
    #
    # plt.plot(x_data, y_data, "b--", label='cos')
    # plt.legend(loc="upper left", shadow=True)
    # plt.pause(0.01)
    plt.ioff()
    plt.show()
    plt.savefig('result_knn_3_150')
    np.savetxt('result_knn_3_150', result)



