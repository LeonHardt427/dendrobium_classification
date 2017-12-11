# -*- coding: utf-8 -*-
# @Time    : 2017/12/9 10:04
# @Author  : LeonHardt
# @File    : bagging_lda_knn.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


if __name__ == '__main__':
    '''
    read data 
    '''
    X = np.loadtxt('x_sample.csv', delimiter=',')
    y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    sc = StandardScaler()
    X_train = sc.fit_transform(X)
    '''
    LDA
    '''
    for component in range(10, 15, 1):
        lda = LinearDiscriminantAnalysis(n_components=component)
        X_train_lda = lda.fit_transform(X_train, y)

        '''
        knn
        '''
        for neighbor in range(1, 6, 1):
            knn = KNeighborsClassifier(n_neighbors=neighbor, p=2, metric='minkowski')

            result = []
            # pic = plt.figure(figsize=(8, 6), dpi=80)
            # plt.ion()
            for time in range(1, 100, 1):
                plt.cla()
                print('the %d times' % time)
                bag = BaggingClassifier(base_estimator=knn, n_estimators=time, max_samples=1.0, max_features=1.0,
                                        bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
                scores = cross_val_score(estimator=bag, X=X_train_lda, y=y, cv=10, n_jobs=-1)

                result.append([time, np.mean(scores), np.std(scores)])

            name = 'bagging_lda' + str(int(component)) + '_K' + str(int(neighbor)) + '.txt'
            np.savetxt(name, result, delimiter=',')

            # '''
            # time_plot
            # '''
            #     if time > 2:
            #         result_plot = np.array(result)
            #         x_data = result_plot[:, 0]
            #         y_data = result_plot[:, 1]
            #
            #         plt.title('Bagging_2NN')
            #         plt.grid(True)
            #         plt.xlabel('Accuracy')
            #         plt.xlim(0, 170)
            #         plt.ylabel('Times')
            #         plt.ylim(0, 1)
            #
            #         plt.plot(x_data, y_data, "b--", label='cos')
            #         plt.legend(loc="upper left", shadow=True)
            #         plt.pause(0.001)
            #
            # plt.ioff()
            # plt.show()
            # plt.savefig('result_lda_knn_3_100')
            # np.savetxt('result_lda_knn_3_100', result)
