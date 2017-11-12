#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/19 下午3:07
# @Author  : LeonHardt
# @File    : train_test_preparation.py

"""
Function: Take all samples into one array, and make them standard. Make the label array.

Data_From: ./Dendrobium_Standard_Data/*.cvs
Save_To: ./
x_sample: feature_matrix
------------------------------
         'F_max', 'F_int', 'SR_1_max', 'SR_1_min', 'SR_10_max', 'SR_10_min', 'SR_100_max', 'SR_100_min'
Sample 1
Sample 2
'''
Sample x
------------------------------

y_label: [1, 1, 1, ..., n, n, n}

"""
import os
import glob
import pandas as pd
import numpy as np

'''
Set parameters of the E-nose data
'''
CATEGORY_NUMBER = 10           # The number of category
SAMPLE_NUMBER_EVERY_CATEGORY = 48      # The number of sample in each category


sample_dir = os.getcwd()
sample_list = glob.glob(sample_dir + '/Dendrobium_Feature_Data/' + '*.csv')
sample_number = 0

'''
Take the features of every sample and stack them into x_sample matrix. Named 'x_sample'
'''
for sample in sample_list:
    df_sample = pd.read_csv(sample)
    np_sample_array = df_sample.values[:, 1:]
    np_feature = np_sample_array.reshape(1, 128)
    if sample_number is 0:
        x_sample = np_feature
    else:
        x_sample = np.vstack((x_sample, np_feature))
    print(str(sample_number) + ' is stacked')
    sample_number += 1

np.savetxt('x_sample.csv', x_sample, delimiter=',')
print('\nx_sample is done \n')
print('There are ' + str(x_sample.shape[0]) + ' samples in this prediction \n')

'''
Make the label-matrix. Named 'y_label'
'''
for category_number in range(CATEGORY_NUMBER):
    if category_number is 0:
        y_label = np.zeros((SAMPLE_NUMBER_EVERY_CATEGORY, 1), dtype='int8')
    else:
        category_label = np.ones((SAMPLE_NUMBER_EVERY_CATEGORY, 1), dtype='int8') * category_number
        y_label = np.vstack((y_label, category_label))

np.savetxt('y_label.csv', y_label, delimiter=',')
print('There are ' + str(CATEGORY_NUMBER) + ' categories in this prediction and each category has ' +
      str(SAMPLE_NUMBER_EVERY_CATEGORY) + ' samples')
print(y_label.shape)





