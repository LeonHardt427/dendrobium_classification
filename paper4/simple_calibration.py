#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/14 下午6:06
# @Author  : LeonHardt
# @File    : simple_calibration.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd() + '/simple_test/'
file_list = glob.glob(path+'*.txt')

for file in file_list:
    name = file.split('/')[-1].split('.')[0].split('_')[1]
    df = pd.read_csv(file)
    data = df['Accuracy'].values
    print(data)
    save_path = path + 'data/'
    np.savetxt(save_path+name+'.txt', data, delimiter=',')

