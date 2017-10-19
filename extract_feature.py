#!usr/bin/python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/20 下午3:07
# @Author  : LeonHardt
# @File    : train_test_preparation.py

"""
Function: Extract features from standard data. 8 features from every sensors, the total number is 128.

Data_From: ./Dendrobium_Standard_Data/*.txt
Save_To: .//Dendrobium_Feature_Data/*.cvs
Feature_Structure:
------------------------------
         'F_max', 'F_int', 'SR_1_max', 'SR_1_min', 'SR_10_max', 'SR_10_min', 'SR_100_max', 'SR_100_min'
Sensor 1
Sensor 2
'''
Sensor 16
------------------------------
"""

import os
import pandas as pd
import numpy as np
import glob

'''
Set parameters of the E-nose data
'''
OPTION_TIME_POINT = 33800
OPTION_SENSOR_NUMBER = 16
OPTION_VOLTAGET_MAX = 3.3
OPTION_R_REF = 3
OPTION_BASELINE_POINT = 2000
OPTION_SR = 1000

'''
Get the direction and the text-name list
'''
data_dir = os.getcwd()
file_name_list = glob.glob(data_dir + '/Dendrobium_Standard_Data/' + '*.txt')

'''
Get the data, extract the 33800 data we will analysis,and sort them as the name of the sensors.
Using DataFrame to do the work
'''

for txt_name in file_name_list:
    df = pd.read_table(txt_name, sep='\s+', header=None)
    df_data = df.iloc[0:33800, 2:18]
    df_data.rename(columns={2: 800, 3: 830, 4: 832, 5: 813, 6: 821, 7: 813, 8: 822, 9: 2600, 10: 826, 11: 880,
                            12: 822, 13: 816, 14: 2602, 15: 2442, 16: 2610, 17: 2611}, inplace=True)
    df_data = df_data.sort_index(axis=1)

    '''
    Minus baseline
    '''
    df_baseline = df_data.iloc[0:OPTION_BASELINE_POINT, :].mean()

    for num_sensor in range(OPTION_SENSOR_NUMBER):
        df_data.iloc[:, num_sensor] = df_data.iloc[:, num_sensor] - df_baseline.values[num_sensor]

    df_data_standard = df_data.drop(range(0, OPTION_BASELINE_POINT))
    df_data_standard.plot()

    '''
    Extract features:
    1. Max
    2. Int
    3-8. Move average of SR, SR*10, SR*100
    '''
    data_feature = pd.DataFrame(columns=['F_max', 'F_int', 'SR_1_max', 'SR_1_min', 'SR_10_max',
                                         'SR_10_min', 'SR_100_max', 'SR_100_min'])

    df_data_standard *= 1000

    data_feature['F_max'] = df_data_standard.max(axis=0)  # The maximum value

    data_feature['F_int'] = df_data_standard.sum(axis=0)  # The integral value

    r_SR = (1 / (OPTION_SR * 1))                     # The move average of 1/(SR*1)
    move_average = np.zeros(df_data_standard.shape)
    move_average[0, :] = r_SR * df_data_standard.iloc[0, :]
    for SR_index in range(1, df_data_standard.shape[1]):
        move_average[SR_index, :] = (1 - r_SR) * df_data_standard.iloc[SR_index - 1, :] + \
                                    r_SR * (df_data_standard.iloc[SR_index, :] - df_data_standard.iloc[SR_index - 1, :])

    data_feature['SR_1_max'] = move_average.max(axis=0)
    data_feature['SR_1_min'] = move_average.min(axis=0)

    r_SR = (1 / (OPTION_SR * 10))  # The move average of 1/(SR*10)
    move_average = np.zeros(df_data_standard.shape)
    move_average[0, :] = r_SR * df_data_standard.iloc[0, :]
    for SR_index in range(1, df_data_standard.shape[1]):
        move_average[SR_index, :] = (1 - r_SR) * df_data_standard.iloc[SR_index - 1, :] + \
                                    r_SR * (df_data_standard.iloc[SR_index, :] - df_data_standard.iloc[SR_index - 1, :])

    data_feature['SR_10_max'] = move_average.max(axis=0)
    data_feature['SR_10_min'] = move_average.min(axis=0)

    r_SR = (1 / (OPTION_SR * 100))  # The move average of 1/(SR*100)
    move_average = np.zeros(df_data_standard.shape)
    move_average[0, :] = r_SR * df_data_standard.iloc[0, :]
    for SR_index in range(1, df_data_standard.shape[1]):
        move_average[SR_index, :] = (1 - r_SR) * df_data_standard.iloc[SR_index - 1, :] + \
                                    r_SR * (df_data_standard.iloc[SR_index, :] - df_data_standard.iloc[SR_index - 1, :])

    data_feature['SR_100_max'] = move_average.max(axis=0)
    data_feature['SR_100_min'] = move_average.min(axis=0)

    '''
    Save the feature to '/Dendrobium_Feature_Data/'
    Format: .csv
    '''
    if os.path.exists(data_dir + '/Dendrobium_Feature_Data/') is not True:
        os.makedirs(data_dir + '/Dendrobium_Feature_Data/')
    file_name_whole = txt_name.split('/')[-1]
    file_name_standard = file_name_whole.split('-')[0] + '-' + file_name_whole.split('-')[1]
    data_feature.to_csv(data_dir + '/Dendrobium_Feature_Data/' + file_name_standard + '.csv')
    print(file_name_standard + ' is OK')
