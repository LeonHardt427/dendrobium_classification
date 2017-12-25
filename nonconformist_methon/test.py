# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 18:29
# @Author  : LeonHardt
# @File    : test.py


import numpy as np

y = np.loadtxt('y_label.csv', delimiter=',', dtype='int8')

labels, y = np.unique(y, return_inverse=True)

correct = np.ones((y.size,), dtype=bool)


sum_label = np.sum(correct)
print(sum_label)
# for i, y_ in enumerate(y):
#
#     correct[i] = prediction[i, int(y_)]