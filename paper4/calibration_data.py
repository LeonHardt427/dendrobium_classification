#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/12 上午10:03
# @Author  : LeonHardt
# @File    : calibration_data.py

import os
import glob

import numpy as np
import pandas as pd


def calibration_data(file_path, save_path):
    """

    Parameters
    ----------
    file_path: path of the file
    save_path: path to save

    Returns
    -------
      None
    """
    file_list = glob.glob(file_path)
    calibration = []
    for k, file in enumerate(file_list):
        df = pd.read_csv(file)
        data = df['Accuracy'].values
        if k == 0:
            calibration = data
        else:
            calibration = np.vstack((calibration, data))
    calibration_mean = np.mean(calibration, axis=0)
    np.savetxt(save_path, calibration_mean, delimiter=',')
    print(calibration_mean)


if __name__ == '__main__':
    names = ['ACP-RandomSubSampler', 'ACP-CrossSampler', 'ACP-BootstrapSampler', 'CCP', 'BCP']
    # methods = ['RF', 'RF(500)', 'SVM(40,0.05)', 'SVM(1000,0.05)', 'SVM(6000,0.001)']
    # methods = ['SVM(6000,0.001)']
    methods = ['SVM(6000,0.001)']
    # names = ['ACP-BootstrapSampler']
    save_path = os.getcwd() + '/calibration/time/'
    for method in methods:
        for name in names:
            print(name)
            path = os.getcwd() + '/summary/time/' + method + '/' + name + '/*.csv'
            save_path = os.getcwd() + '/calibration/time/' + method + '/'
            if os.path.exists(save_path) is not True:
                os.makedirs(save_path)
            save_file = save_path + name + '.txt'
            calibration_data(path, save_file)
