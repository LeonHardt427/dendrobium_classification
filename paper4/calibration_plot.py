#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/14 上午10:12
# @Author  : LeonHardt
# @File    : calibration_plot.py

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Form of calibration data: numpy.array
"""


def calibration_plot(file_path, save_path, save_option=1):
    """

    Parameters
    ----------
    file_path: str
            calibration data path
    save_path: str
            save picture path
    save_option : int
            1   save
            0   not save
    Returns
    -------
        None
    """
    sig = np.arange(0, 1.0001, 0.005)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)

    ax = [ax1, ax2, ax3, ax4, ax5]
    file_list = glob.glob(file_path)
    for k, file in enumerate(file_list):
        name = file.split('/')[-1].split('.')[0]
        ax_temp = ax[k]
        data = np.loadtxt(file, delimiter=',')

        ax_temp.plot(sig, data, linestyle='-', color='red', label=name)
        ax_temp.plot(sig, sig, linestyle='--', color='black')

        ax_temp.set_xlabel("Significance", fontsize=10, fontweight='bold')
        ax_temp.set_ylabel("Error rate", fontsize=10, fontweight='bold')
        ax_temp.set_xlim(0, 1)
        ax_temp.set_ylim(0, 1)
        # ax_temp.set_xticklabels(fontsize=8, fontweight='bold')
        # ax_temp.set_yticklabels(fontsize=8, fontweight='bold')
        ax_temp.legend(loc='best')
    plt.show()
    if save_option == 1:
        fig.savefig(save_path)

if __name__ == '__main__':
    method = 'simple_test'
    # path = os.getcwd()+'/calibration/'+method+'/*.txt'
    path = os.getcwd() + '/simple_test/data/*.txt'
    save_path = os.getcwd() + '/calibration_plot/'
    if os.path.exists(save_path) is not True:
        os.makedirs(save_path)
    save_file = save_path+method+'.png'
    calibration_plot(path, save_file, save_option=0)

