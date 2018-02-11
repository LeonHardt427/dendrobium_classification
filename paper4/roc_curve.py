# -*- coding: utf-8 -*-
# @Time    : 2018/1/19 14:19
# @Author  : LeonHardt
# @File    : roc_curve.py

print(__doc__)

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


frameworks = ['BCP', 'ICP', 'CP']
# methods = ['1NN', 'SVM', 'Tree']
methods = ['Tree']
# -------------------------------------------------------------
# stack data
# frameworks = ['BCP', 'ICP', 'CP']
# methods = ['1NN', 'SVM', 'Tree']
# for framework in frameworks:
#     for method in methods:
#         path = os.getcwd()
#         files = glob.iglob(path+'/force_summary/'+framework+'/'+method+'/*.txt')
#         summary = []
#         for num, file in enumerate(files):
#             data = np.loadtxt(file, delimiter=',')
#             if num == 0:
#                 summary = data
#             else:
#                 summary = np.vstack((summary, data))
#
#             print(summary)
#         save_path=os.getcwd()+'/roc/'+framework+'_'+method+'.txt'
#         if os.path.exists(os.getcwd()+'/roc/') is not True:
#             os.makedirs(os.getcwd()+'/roc/')
#         if os.path.exists(save_path) is True:
#             os.remove(save_path)
#         np.savetxt(save_path, summary, delimiter=',')

# -------------------------------------------------------------
#  stand data
# for method in methods:
#     save_path = os.getcwd() + '/roc/' + method + '/*.txt'
#     files = glob.iglob(save_path)
#     prediction = []
#     prob = []
#     for  index, file in enumerate(files):
#         data = np.loadtxt(file, delimiter=',')
#         locs = data.shape[0]
#         test = data[:, 9]
#         for loc in range(locs):
#             probility = (data[loc, int(data[loc, 9])]) / (np.sum(data[loc, :]) - data[loc, 9])
#             prob.append(probility)
#             p_values = np.array(data[loc, 0:9])
#             correct = (p_values.argmax() == int(test[loc]))
#             prediction.append(correct)
#         save_prob = os.getcwd() + '/roc/' + method + '/summary/'
#         if os.path.exists(save_prob) is not True:
#             os.makedirs(save_prob)
#         np.savetxt(save_prob+frameworks[index]+'.txt', prob, delimiter=',')
#         np.savetxt(save_prob+frameworks[index]+'test.txt', prediction, delimiter=',')

#----------------------------------------------------------------
# roc

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

fig = plt.figure()

for method in methods:
    for index in range(3):
        save_prob = os.getcwd() + '/roc/' + method + '/summary/'
        y = np.loadtxt(save_prob+frameworks[index]+'.txt', delimiter=',')
        test = np.loadtxt(save_prob+frameworks[index]+'test.txt', delimiter=',', dtype='int8')

        fpr, tpr, thresholds = roc_curve(test, y, pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (index, roc_auc))

    # 画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

# mean_tpr /= len(cv)  # 在mean_fpr100个点，每个点处插值插值多次取平均
# mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
# mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
# # 画平均ROC曲线
# # print mean_fpr,len(mean_fpr)
# # print mean_tpr
# plt.plot(mean_fpr, mean_tpr, 'k--',
#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()