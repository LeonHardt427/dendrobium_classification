#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/3 下午3:26
# @Author  : LeonHardt
# @File    : dendrobium_sample_preparation.py

import os
import glob
import numpy as np
import pandas as pd

"""
Set parameter
"""
CATEGORY = 10
SAMPLE_NUMBER_EACH_CATEGORY = 1
DOCUMENT = 'F100toT1000'

"""
Read files
Stack x_sample
"""
dir_data = os.getcwd()
print(dir_data + '/Dendrobium_Time_Feature/' + DOCUMENT + '/*.csv')
sample_name_list = glob.glob(dir_data + '/Dendrobium_Time_Feature/' + DOCUMENT + '/*.csv')
sample_number = 0

for sample in sample_name_list:
    df_sample = pd.read_csv(sample)
    sample_value = df_sample.values[:, 1:]
    feature_number = sample_value.shape[0] * sample_value.shape[1]
    print(feature_number)
    sample_standard = sample_value.reshape(1, feature_number)
    if sample_number is 0:
        x_sample = sample_standard
    else:
        x_sample = np.vstack((x_sample, sample_standard))

    print(str(sample) + ' is stacked')
    sample_number += 1

np.savetxt('x_sample ' + DOCUMENT + '.csv', x_sample, delimiter=',')
print('\nx_sample of' + DOCUMENT + ' is done \n')
print('There are ' + str(x_sample.shape[0]) + ' samples in this prediction \n')

"""
Stack y_sample
"""

for category_number in range(CATEGORY):
    if category_number is 0:
        y_label = np.zeros((SAMPLE_NUMBER_EACH_CATEGORY, 1), dtype='int8')
    else:
        category_label = np.ones((SAMPLE_NUMBER_EACH_CATEGORY, 1), dtype='int8') * category_number
        y_label = np.vstack((y_label, category_label))

np.savetxt('y_label' + DOCUMENT + '.csv', y_label, delimiter=',')
print('There are ' + str(CATEGORY) + ' categories in this prediction and each category has ' +
      str(SAMPLE_NUMBER_EACH_CATEGORY) + ' samples')
print(y_label.shape)


