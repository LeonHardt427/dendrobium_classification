# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 10:49
# @Author  : LeonHardt
# @File    : efficiency_data.py

"""
change sig_error data to efficency data
sig_error data: pd
                |no|sig|accuracy|Average_count|
"""

import os
import glob
import numpy as np
import pandas as pd


def efficiency_data(file_path, save_path):
    file_list = glob.glob(file_path)
    efficiency = []
    for index, file in enumerate(file_list):
        data = pd.read_csv(file)['Average_count'].values
        if index == 0 :
            efficiency = data
            print(efficiency.shape)
        else:
            efficiency = np.vstack((efficiency, data))
            print(efficiency.shape)
    efficiency_mean = np.mean(efficiency, axis=0)
    np.savetxt(save_path, efficiency_mean, delimiter=',')
    print(efficiency_mean)
    print("Done")


if __name__ == '__main__':
    # methods = ['ACP_RF(500)', 'ACP_SVM(60,0.001)', 'ICP_RF(500)', 'ICP_SVM(6000,0.001)']
    methods = ['ICP_RF(500)', 'ICP_SVM(6000,0.001)']
    for method in methods:
        path = os.getcwd() + '/summary/feature/' + method + '/*.csv'
        save_path = os.getcwd() + '/efficiency/feature/'
        if os.path.exists(save_path) is not True:
            os.makedirs(save_path)
        save_file = save_path + method + '.txt'
        efficiency_data(path, save_file)
