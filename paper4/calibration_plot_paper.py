# -*- coding: utf-8 -*-
# @Time    : 2018/3/15 20:29
# @Author  : LeonHardt
# @File    : calibration_plot_paper.py

"""
Standard plot in paper4

"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd()
sig = np.arange(0, 1.0001, 0.005)

fig = plt.figure(figsize=(9, 7))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

data1 = np.loadtxt(path + '/calibration/paper/ICP_SVM(60,0.001).txt')
ax1.plot(sig, data1, linestyle='-', color='red', label='ICP-SVM')
ax1.plot(sig, sig, linestyle='--', color='black')
ax1.set_xlabel("Significance level", fontsize=10, fontweight='bold')
ax1.set_ylabel("Error rate", fontsize=10, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.legend(loc='best')


data2 = np.loadtxt(path + '/calibration/paper/ACP_SVM(60,0.001).txt')
ax2.plot(sig, data2, linestyle='-', color='red', label='ACP-SVM')
ax2.plot(sig, sig, linestyle='--', color='black')
ax2.set_xlabel("Significance level", fontsize=10, fontweight='bold')
ax2.set_ylabel("Error rate", fontsize=10, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend(loc='best')


data3 = np.loadtxt(path + '/calibration/paper/ICP_RF(500).txt')
ax3.plot(sig, data3, linestyle='-', color='red', label='ICP-RF')
ax3.plot(sig, sig, linestyle='--', color='black')
ax3.set_xlabel("Significance level", fontsize=10, fontweight='bold')
ax3.set_ylabel("Error rate", fontsize=10, fontweight='bold')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.legend(loc='best')

data4 = np.loadtxt(path + '/calibration/paper/ACP_RF(500).txt')
ax4.plot(sig, data4, linestyle='-', color='red', label='ACP-RF')
ax4.plot(sig, sig, linestyle='--', color='black')
ax4.set_xlabel("Significance level", fontsize=10, fontweight='bold')
ax4.set_ylabel("Error rate", fontsize=10, fontweight='bold')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.legend(loc='best')

plt.show()


#
# ax = [ax1, ax2, ax3, ax4]
# file_list = glob.glob(file_path)
# for k, file in enumerate(file_list):
#     name = file.split('\\')[-1].split('.')[0]
#     ax_temp = ax[k]
#     data = np.loadtxt(file, delimiter=',')
#
#     ax_temp.plot(sig, data, linestyle='-', color='red', label=name)
#     ax_temp.plot(sig, sig, linestyle='--', color='black')
#
#     ax_temp.set_xlabel("Significance", fontsize=10, fontweight='bold')
#     ax_temp.set_ylabel("Error rate", fontsize=10, fontweight='bold')
#     ax_temp.set_xlim(0, 1)
#     ax_temp.set_ylim(0, 1)
#     # ax_temp.set_xticklabels(fontsize=8, fontweight='bold')
#     # ax_temp.set_yticklabels(fontsize=8, fontweight='bold')
#     ax_temp.legend(loc='best')
# plt.show()
# if save_option == 1:
#     fig.savefig(save_path)