# -*- coding: utf-8 -*-
# @Time    : 2018/3/19 19:24
# @Author  : LeonHardt
# @File    : confidence_credibility.py

import os
import glob
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory, ClassifierNc
from nonconformist.evaluation import class_avg_c, class_mean_errors


# frameworks = ['ACP_RF(500)', 'ACP_SVM(60,0.001)', 'ICP_RF(500)', 'ICP_SVM(60,0.001)']
# for index, framework in enumerate(frameworks):
#     save_prob = os.getcwd() + '/prediction/paper/' + framework + '.txt'
#     data = np.loadtxt(save_prob, delimiter=',')
#     data_temp = np.sort(np.array(data[:, 0:9]))
#     conf = 1 - data_temp[:, -2]
#     cred = data_temp[:, -1]
#     print(framework+'_'+':')
#     print('conf is %.4f std is %.4f ' % (np.mean(conf), np.std(conf)))
#     print('cred is %.4f std is %.4f ' % (np.mean(cred), np.std(cred)))
#     # for i in range(data_temp.shape[0]):
#     #     conf_temp = max(data_temp[i, :])

file = 'ACP_SVM(60,0.001)/'
path = os.getcwd() + '/prediction/' + file + '*.txt'

predictions = glob.glob(path)
for num, prediction in enumerate(predictions):
    name = prediction.split('\\')[-1].split('.')[0]
    print(name)
    data = np.loadtxt(prediction, delimiter=',')
    data_temp = np.sort(np.array(data[:, 0:9]))
    conf = 1 - data_temp[:, -2]
    cred = data_temp[:, -1]
    print('conf is %.4f std is %.4f ' % (np.mean(conf), np.std(conf)))
    print('cred is %.4f std is %.4f ' % (np.mean(cred), np.std(cred)))



