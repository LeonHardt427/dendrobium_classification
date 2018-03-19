# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 15:35
# @Author  : LeonHardt
# @File    : efficiency_plot_paper.py

"""
Standard efficiency plot in paper4

"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd()
sig = np.arange(0, 1.0001, 0.005)

fig = plt.figure(figsize=(9, 7))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

data1 = np.loadtxt(path + '/efficiency/time/ICP_SVM(60,0.001).txt')
ax1.plot(sig, data1, linestyle='-', color='red', label='ICP_SVM')
data2 = np.loadtxt(path + '/efficiency/time/ACP_SVM(60,0.001).txt')
ax1.plot(sig, data2, linestyle='-', color='blue', label='ACP_SVM')
ax1.plot(sig, np.ones(sig.shape), linestyle='--', color='grey')
ax1.set_xlabel("Significance level", fontsize=10, fontweight='bold')
ax1.set_ylabel("Average size", fontsize=10, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 10)
ax1.legend(loc='best')

data3 = np.loadtxt(path + '/efficiency/time/ICP_RF(500).txt')
ax2.plot(sig, data3, linestyle='-', color='red', label='ICP_RF')
data4 = np.loadtxt(path + '/efficiency/time/ACP_RF(500).txt')
ax2.plot(sig, data4, linestyle='-', color='blue', label='ACP_RF')
ax2.plot(sig, np.ones(sig.shape), linestyle='--', color='grey')
ax2.set_xlabel("Significance level", fontsize=10, fontweight='bold')
ax2.set_ylabel("Average size", fontsize=10, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 10)
ax2.legend(loc='best')


plt.show()

