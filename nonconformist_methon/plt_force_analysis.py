# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 16:28
# @Author  : LeonHardt
# @File    : plt_force_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd()

# -------------------------------------------------------------------------
# without lda
# -------------------------
# get all.csv sheet
# file_path = path + '/summary_force/'
# if os.path.exists(file_path + 'all.csv') is True:
#     summary_force = pd.DataFrame.from_csv(file_path + 'all.csv')
#     print(summary_force)
# else:
#     index_name = ['1NN', '3NN', 'Tree', 'RandomForest', 'SVM']
#     icp_summary = pd.Series.from_csv(file_path + 'icp_force.csv')
#     icp_summary.index.values[:] = index_name
#     bcp_summary = pd.Series.from_csv(file_path + 'bcp_force.csv')
#     bcp_summary.index.values[:] = index_name
#     ccp_summary = pd.Series.from_csv(file_path + 'ccp_force.csv')
#     ccp_summary.index.values[:] = index_name
#     # print(icp_summary)
#     # print(bcp_summary)
#     # print(ccp_summary)
#     summary_force = pd.DataFrame([icp_summary, bcp_summary, ccp_summary], index=['icp', 'bcp', 'ccp'])
#     summary_force.to_csv(file_path + 'all.csv')
#     print(summary_force)

# ------------------------------------------------------
# make plot

# fig = plt.figure()
# n_group = np.arange(summary_force.shape[1])
# bar_with = 0.2
# ticks = ['1NN', '3NN', 'Tree', 'RandomForest', 'SVM']
#
# plt.bar(n_group, summary_force.ix[0, :].values, bar_with, label='icp', color='b')
# plt.bar(n_group+bar_with, summary_force.ix[1, :].values, bar_with, label='bcp', color='g')
# plt.bar(n_group+bar_with*2, summary_force.ix[2, :].values, bar_with, label='ccp', color='coral')
# plt.legend()
# plt.xlabel('classify_method')
# plt.ylabel('accuracy')
# plt.xticks(n_group+bar_with, ticks)
#
# plt.show()

# -------------------------------------------------------------------------
# without lda
# -------------------------
# get all.csv sheet

file_path = path + '/summary_force/lda/'
if os.path.exists(file_path + 'all_lda.csv') is True:
    summary_force = pd.DataFrame.from_csv(file_path + 'all_lda.csv')
    print(summary_force)
else:
    index_name = ['1NN', '3NN', 'Tree', 'RandomForest', 'SVM']
    icp_summary = pd.Series.from_csv(file_path + 'icp_lda_force.csv')
    icp_summary.index.values[:] = index_name
    bcp_summary = pd.Series.from_csv(file_path + 'bcp_lda_force.csv')
    bcp_summary.index.values[:] = index_name
    ccp_summary = pd.Series.from_csv(file_path + 'ccp_lda_force.csv')
    ccp_summary.index.values[:] = index_name
    # print(icp_summary)
    # print(bcp_summary)
    # print(ccp_summary)
    summary_force = pd.DataFrame([icp_summary, bcp_summary, ccp_summary], index=['icp_lda', 'bcp_lda', 'ccp_lda'])
    summary_force.to_csv(file_path + 'all_lda.csv')
    print(summary_force)

# ----------------------------------------------------------------------
# make plot

fig = plt.figure()
n_group = np.arange(summary_force.shape[1])
bar_with = 0.2
ticks = ['1NN', '3NN', 'Tree', 'RandomForest', 'SVM']

plt.bar(n_group, summary_force.ix[0, :].values, bar_with, label='icp_lda', color='b')
plt.bar(n_group+bar_with, summary_force.ix[1, :].values, bar_with, label='bcp_lda', color='g')
plt.bar(n_group+bar_with*2, summary_force.ix[2, :].values, bar_with, label='ccp_lda', color='coral')
plt.legend()
plt.xlabel('classify_method')
plt.ylabel('accuracy')
plt.xticks(n_group+bar_with, ticks)

plt.show()
