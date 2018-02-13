# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 14:06
# @Author  : LeonHardt
# @File    : force_prediction.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory, ClassifierNc
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier

from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.acp import BootstrapConformalClassifier
from force_value import force_mean_errors

# ----------------------------------------
# preprocessing
# -----------------------------------------
path = os.getcwd()

X = np.loadtxt('ginseng_x_sample.txt', delimiter=',')
y = np.loadtxt('ginseng_y_label.txt', delimiter=',')

sc = StandardScaler()
X = sc.fit_transform(X)

# --------------------------------------------
# prediction
# --------------------------------------------

summary = []

# simple_model = KNeighborsClassifier(n_neighbors=3)
# model_name = '3NN'
#
simple_model = RandomForestClassifier(n_estimators=50, criterion='entropy')
model_name = "Tree"

# simple_model = KNeighborsClassifier(n_neighbors=1)
# model_name = '1NN'

# simple_model = SVC(C=40.0, gamma=0.005, probability=True)
# model_name = "SVM"

# -----------------------------------------------------------------------------
# Define models
# -----------------------------------------------------------------------------

models = {  'ACP-RandomSubSampler'  : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model))),
                                        RandomSubSampler()),
            'ACP-CrossSampler'      : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model))),
                                        CrossSampler()),
            'ACP-BootstrapSampler'  : AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model))),
                                        BootstrapSampler()),
            'CCP'                   : CrossConformalClassifier(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model)))),
            'BCP'                   : BootstrapConformalClassifier(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(simple_model)))),
          }

framework_name = 'BCP'

error_summary = []
result_summary = []

# --------------------------------------------------------------------------------------------
# force_prediction
save_path = os.getcwd()+'/force_summary/' + framework_name+'/'+model_name+'/'
if os.path.exists(save_path) is not True:
    os.makedirs(save_path)

s_folder = StratifiedKFold(n_splits=10, shuffle=True)

for index, (train, test) in enumerate(s_folder.split(X, y)):
    x_train, x_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    truth = y_test.reshape((-1, 1))
    # -----------------------------------------------
    # BCP

    conformal_model = BootstrapConformalClassifier(IcpClassifier(ClassifierNc(ClassifierAdapter(simple_model))),
                                                   n_models=10)
    conformal_model.fit(x_train, y_train)

    # ------------------------------------------
    # ICP
    # x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(x_train, y_train, test_size=0.3, shuffle=True,
    #                                                         random_state=1)
    # nc = NcFactory.create_nc(model=simple_model)
    # conformal_model = IcpClassifier(nc)
    # conformal_model.fit(x_train_sp, y_train_sp)
    # conformal_model.calibrate(x_cal, y_cal)

    # ---------------------------------------------------
    # CP
    # nc = NcFactory.create_nc(model=simple_model)
    # conformal_model = IcpClassifier(nc)
    # conformal_model.fit(x_train, y_train)
    # conformal_model.calibrate(x_train, y_train)

    prediction = conformal_model.predict(x_test, significance=None)
    table = np.hstack((prediction, truth))
    result = [1 - force_mean_errors(prediction, truth)]
    np.savetxt(save_path + '/forece_'+model_name +'_'+framework_name+str(index)+'.txt', table, delimiter=',')
    if index == 0:
        result_summary = result
    else:
        result_summary = np.vstack((result_summary, result))

    if np.unique(y_test).shape[0] == 9:
        print('True')
    else:
        print('Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Accuracy: {}'.format(result[0]))

save_file = save_path + 'significance_error_'+model_name +'_'+framework_name+'10.txt'
if os.path.exists(save_file):
    os.remove(save_file)
np.savetxt(save_file, result_summary, delimiter=',')
df_summary = pd.DataFrame(result_summary, columns=['Accuracy'])
print(df_summary['Accuracy'].mean())

