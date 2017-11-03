#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/3 上午10:42
# @Author  : LeonHardt
# @File    : dendrobium_time_extract.py

import os
import numpy as np
import glob
import pandas as pd

"""
Set parameter
"""
OPTION_SENSOR_NUMBER = 16
TIME_START = 0
TIME_END = 1000
POINT_START = TIME_START * 100
POINT_END =  TIME_END * 100
FREQUENCY = 100
FEATURE_NUMBER = (POINT_END - POINT_START) / FREQUENCY
OPTION_BASELINE = 2000
TIME = 33800

"""
Get direction
"""

data_dir = os.getcwd()
file_name_list = glob.glob(data_dir + '/Dendrobium_Original_Data/' + '*.txt')

"""
Extract time feature
"""
for file_name in file_name_list:
    df = pd.read_table(file_name, sep='\s+', header=None)
    df_data = df.iloc[0:33800, 2:18]
    df_data.rename(columns={2: 800, 3: 830, 4: 832, 5: 813, 6: 821, 7: 813, 8: 822, 9: 2600, 10: 826, 11: 880,
                            12: 822, 13: 816, 14: 2602, 15: 2442, 16: 2610, 17: 2611}, inplace=True)
    df_data = df_data.sort_index(axis=1)

    # Minus baseline
    data_baseline = df_data.iloc[0:OPTION_BASELINE, :].mean()  # 存疑
    for sensor_number in range(OPTION_SENSOR_NUMBER):
        df_data.iloc[:, sensor_number] = df_data.iloc[:, sensor_number] - data_baseline.values[sensor_number]

    df_standard = df_data.drop(range(0, OPTION_BASELINE))
    df_standard.plot()

    data_time_feature = df_standard.iloc[range(0, min(POINT_END, df_standard.shape[0]), FREQUENCY)] * 100
    """
    Save time feature
    To: './Dendrobium_Time_Feature/' + Frequency
    """

    if os.path.exists(data_dir + '/Dendrobium_Time_Feature/F' + str(FREQUENCY) + 'toT' + str(TIME_END)) is not True:
        os.makedirs(data_dir + '/Dendrobium_Time_Feature/F' + str(FREQUENCY) + 'toT' + str(TIME_END))

    file_name_whole = file_name.split('/')[-1]
    file_name_standard = file_name_whole.split('-')[0] + '-' + file_name_whole.split('-')[1]
    data_time_feature.to_csv(
        data_dir + '/Dendrobium_Time_Feature/F' + str(FREQUENCY) + 'toT' + str(TIME_END) + '/' +
        file_name_standard + ".csv")
    print(data_time_feature.shape)
    print(file_name_standard + ' is OK')

if __name__ == '__main__':
    print('all are done')

