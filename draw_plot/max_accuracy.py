# -*- coding: utf-8 -*-
# @Time    : 2017/12/14 19:45
# @Author  : LeonHardt
# @File    : max_accuracy.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# FILE_NAME = '/bagging_lda_sample0.8_knn'
FILE_NAME = '/bagging_ldaright_knn'

if __name__ == '__main__':
    dir = os.getcwd()
    dir = dir + '/draw_data' + FILE_NAME +'/result/'
    name_list = glob.glob(dir+'*.txt')
    for file_name in name_list:
        result = np.loadtxt(file_name, delimiter=',')
        result_accuracy = result[:, 1]
        result_iteration = result[:, 0]
        print('the max accuracy is ' + str(result_accuracy.max()))
        print('the bagging time is ' + str(result_iteration[np.argmax(result_accuracy)]))






