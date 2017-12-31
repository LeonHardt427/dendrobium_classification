# -*- coding: utf-8 -*-
# @Time    : 2017/12/29 15:27
# @Author  : LeonHardt
# @File    : bcp_ida_classification.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory, ClassifierNc
from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.acp import BootstrapConformalClassifier

# ----------------------------------------
# preprocessing
# -----------------------------------------
path = os.getcwd()

X = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_label.csv', delimiter=',')

# --------------------------------------------
# prediction
# --------------------------------------------

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # split sample

significance = 0.22
summary = []

# model = {'SVM': SVC(C=4000, kernel='rbf', gamma=0.001, probability=True),
#          'RandomForest': RandomForestClassifier(n_estimators=12, criterion='entropy'),
#          'Tree': DecisionTreeClassifier(criterion='entropy', max_depth=6),
#          '1NN': KNeighborsClassifier(n_neighbors=1),
#          '3NN': KNeighborsClassifier(n_neighbors=3)
#          }

simple_model = SVC(C=4000, kernel='rbf', gamma=0.001, probability=True)
model_name = 'SVM'

sc = StandardScaler()
X = sc.fit_transform(X)

# simple_model = RandomForestClassifier(n_estimators=12, criterion='entropy')
# model_name = 'RandomForest'

# simple_model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
# model_name = 'Tree'

# simple_model = KNeighborsClassifier(n_neighbors=3)
# model_name = '3NN'

# simple_model = KNeighborsClassifier(n_neighbors=1)
# model_name = '1NN'

s_folder = StratifiedKFold(n_splits=10, shuffle=True)
for k, (train, test) in enumerate(s_folder.split(X, y)):
    x_train_std, x_test_std = X[train], X[test]
    y_train, y_test = y[train], y[test]
    truth = y_test.reshape((-1, 1))

    lda = LinearDiscriminantAnalysis(n_components=9)
    x_train_lda = lda.fit_transform(x_train_std, y_train)
    x_test_lda = lda.transform(x_test_std)

    nc_fun = NcFactory.create_nc(model=simple_model)
    model = BootstrapConformalClassifier(IcpClassifier(nc_fun))
    model.fit(x_train_lda, y_train)
    prediction = model.predict(x_test_lda, significance=None)
    table = np.hstack((prediction, truth))
    result = [1 - class_mean_errors(prediction, truth, significance=significance),
              class_avg_c(prediction, truth, significance=significance)]
    if k == 0:
        summary = result
    else:
        summary = np.vstack((summary, result))
    print('\nBCP-LDA')
    print('Accuracy: {}'.format(result[0]))
    print('Average count: {}'.format(result[1]))

df_summary = pd.DataFrame(summary, columns=['Accuracy', 'Average_count'])
print(df_summary)
summary_path = path + '/summary/bcp/lda/'

summary_file = summary_path + 'bcp_lda_' + model_name + '.csv'
if os.path.exists(summary_path) is not True:
    os.makedirs(summary_path)
if os.path.exists(summary_file) is True:
    os.remove(summary_file)

df_summary.to_csv(summary_file)
