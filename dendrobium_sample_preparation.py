#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/3 下午3:26
# @Author  : LeonHardt
# @File    : dendrobium_sample_preparation.py

"""
Parameter need to change:
SAMPLE_NUMBER_EACH_CATEGORY;
DOCUMENT;
SAMPLE_SORT;
"""


import os
import glob
import numpy as np
import pandas as pd

"""
Set parameter
"""
CATEGORY = 10
DOCUMENT = 'F500toT159'

# SAMPLE_NUMBER_EACH_CATEGORY = 10   # test
# SAMPLE_SORT = 'test'

SAMPLE_NUMBER_EACH_CATEGORY = 38   # train
SAMPLE_SORT = 'train'
"""
Read files
Stack x_sample
"""
dir_data = os.getcwd()
print(dir_data + '/Dendrobium_Time_Feature/Size02/' + DOCUMENT + '/' + SAMPLE_SORT + '/*.csv')
sample_name_list = glob.glob(dir_data + '/Dendrobium_Time_Feature/Size02/' + DOCUMENT + '/' + SAMPLE_SORT + '/*.csv')
sample_number = 0

for sample in sample_name_list:
    os.chdir(os.path.dirname(sample))
    df_sample = pd.read_csv(os.path.basename(sample))
    os.chdir(dir_data)
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

np.savetxt('x_sample_' + SAMPLE_SORT +'_02' + DOCUMENT + '.csv', x_sample, delimiter=',')
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

np.savetxt('y_label_' + SAMPLE_SORT + '_02' + DOCUMENT + '.csv', y_label, delimiter=',')
print('There are ' + str(CATEGORY) + ' categories in this prediction and each category has ' +
      str(SAMPLE_NUMBER_EACH_CATEGORY) + ' samples')
print(x_sample.shape)
print(y_label.shape)


