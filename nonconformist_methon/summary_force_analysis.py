# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 14:15
# @Author  : LeonHardt
# @File    : summary_force_analysis.py

import os
import glob

import pandas as pd
# ----------------------------------------
# without lda
#
# path = os.getcwd()
# save_path = path + '/summary_force/'
# if os.path.exists(save_path) is not True:
#     os.makedirs(save_path)
# file_list = ['icp', 'bcp', 'ccp']
#
# for file in file_list:
#     file_path = path + '/summary/' + file + '/force/'
#     summary_file_list = glob.iglob(file_path + '*.csv')
#     dic = {}
#     for index, summary_file in enumerate(summary_file_list):
#         summary = pd.read_csv(summary_file)
#         name = summary_file.split('\\')[-1].split('.')[0]
#         accuracy = summary.mean(axis=0)['Accuracy']
#         dic[name] = accuracy
#     # print(dic)
#     index_name = [file+'_1NN', file+'_3NN', file+'_Tree', file+'_RandomForest', file+'_SVM']
#     result = pd.Series(dic, index=index_name)
#     # result = pd.Series(dic)
#     if os.path.exists(save_path + file + '_force.csv') is True:
#         os.remove(save_path + file + '_force.csv')
#     print(result)
#     result.to_csv(save_path + file + '_force.csv')

# ----------------------------------------
# with lda

path = os.getcwd()
save_path = path + '/summary_force/lda/'
if os.path.exists(save_path) is not True:
    os.makedirs(save_path)
file_list = ['icp', 'bcp', 'ccp']

for file in file_list:
    file_path = path + '/summary/' + file + '/lda_force/'
    summary_file_list = glob.iglob(file_path + '*.csv')
    dic = {}
    for index, summary_file in enumerate(summary_file_list):
        summary = pd.read_csv(summary_file)
        name = summary_file.split('\\')[-1].split('.')[0]
        accuracy = summary.mean(axis=0)['Accuracy']
        dic[name] = accuracy

    print(dic)
    index_name = [file+'_1NN', file+'_3NN', file+'_Tree', file+'_RandomForest', file+'_SVM']
    result = pd.Series(dic, index=index_name)
    # result = pd.Series(dic)
    if os.path.exists(save_path + file + '_lda_force.csv') is True:
        os.remove(save_path + file + '_lda_force.csv')
    print(result)
    result.to_csv(save_path + file + '_lda_force.csv')
