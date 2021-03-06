# -*- coding: utf-8 -*-
# @Time    : 2017/12/9 20:14
# @Author  : LeonHardt
# @File    : plot_bagging_lda_knn.py


import os
import numpy as np
import matplotlib.pyplot as plt


dir_file = os.getcwd()
dir_file = dir_file + '/draw_data/bagging_ldaright_knn/'
KNN = 4

marker = ['.', ',', 'o', 'v', '<', '*', '+', '1', '2']
color = ['red', 'blue', 'cyan', 'yellow', 'black', 'green', 'pink', 'orange', 'magenta', 'lime', 'navy', 'gold',
         'peru', 'grey']

if __name__ == '__main__':
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")        # background line
    ax = plt.gca()
    ax.spines['top'].set_visible(False)   # cancel top ground
    ax.spines['right'].set_visible(False)

    for lda in range(2, 10, 1):
        number = lda
        name_file = dir_file + 'bagging_lda' + str(lda) + '_K' + str(KNN) + '.txt'
        data = np.loadtxt(name_file, delimiter=',')
        X = data[:, 0]
        y = data[:, 1]

        accuracy = np.max(y)
        index = np.argmax(y)
        if lda == 2:
            result = [index, accuracy]
        else:
            now_result = np.array([index, accuracy])
            result = np.vstack((result, now_result))
        print(result)
        print('LDA'+str(lda)+'_'+str(KNN)+'NN:step= %d , accuracy = %.3f' % (index, accuracy))

        plt.plot(X, y, linestyle='-', color=color[number], label='lda'+str(lda), linewidth=1.5)

    np.savetxt('result_'+str(KNN)+'NN_10.txt', X=result, delimiter=',')

    plt.xticks(range(0, 50, 5), fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title('LDA_KNN' + str(KNN), fontsize=12, fontweight='bold')
    plt.xlabel('n_estimators', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.xlim(0, 50)
    plt.legend(loc='lower right', numpoints=1)
    leg = plt.gca().get_legend()
    text = leg.get_texts()
    plt.setp(text, fontsize=12, fontweight='bold')

    plt.savefig('KNN'+str(KNN)+'_10.png', format='png')
    plt.show()







