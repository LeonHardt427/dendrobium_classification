# -*- coding: utf-8 -*-
# @Time    : 2017/12/27 16:08
# @Author  : LeonHardt
# @File    : summary_analysis.py

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# analysis DataFrame data


def analysis_icp(pd_data):
    """

    Parameters
    ----------
    pd_data : str
        the file_path to analysis

    Returns
    -------
    statistice: numpy
        the analyze result : ['mean_error', 'average_count']

    """
    df_data = pd.read_csv(pd_data)
    statistics = df_data.mean(axis=0).values
    return statistics

# -----------------------------
# main


if __name__ == '__main__':
    framework = 'icp'
    path = os.getcwd()
    file_path = path + '/summary/' + framework + '/'
    files = glob.iglob(file_path+'*.csv')
    summary = []
    index = []
    for i, file in enumerate(files):
        file_name = file.split('\\')[-1].split('.')[0]
        if i == 0:
            summary = analysis_icp(file)[1: 3]
            index = [file_name]
        else:
            summary = np.vstack((summary, analysis_icp(file)[1: 3]))
            index.append(file_name)

    summary_all = pd.DataFrame(summary, index=index, columns=['Accuracy', 'average_count'])
    if os.path.exists('summary_all_'+framework+'.csv') is True:
        os.remove('summary_all_'+framework+'.csv')
    summary_all.to_csv('summary_all_'+framework+'.csv')
    print(summary_all)

# --------------------------
# draw plt_bar
    figure = plt.figure()
    n_group = np.arange(len(index))
    bar_with = 0.2
    plt.bar(n_group, summary_all['Accuracy'], bar_with, label="accuracy")
    plt.bar(n_group+bar_with, summary_all['average_count'], bar_with, label="average_count", color='g')
    plt.legend()
    plt.xlabel('classify_method')
    plt.ylabel('value')
    plt.xticks(n_group+bar_with, index)

    plt.show()





