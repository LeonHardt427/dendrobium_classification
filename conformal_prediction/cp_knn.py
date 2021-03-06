#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 上午11:15
# @Author  : LeonHardt
# @File    : cp_knn.py


import numpy as np


def ConformalPredictionKnn(x_train, x_test, y_train, k):
    y_train = y_train.reshape((-1, 1))
    all_label = np.unique(y_train).reshape((-1, 1))
    all_label_number = all_label.size
    p_value = np.zeros((x_test.shape[0], all_label_number))
    sample_number = x_test.shape[0]

    for sample in range(sample_number):
        print(str(k)+'NN, the ' + str(sample) + ' predict is start')
        sample_test = x_test[sample, :]
        predict = np.zeros(all_label_number)

        # stack the train set and the test sample
        for label_number in range(all_label_number):
            y_label = all_label[label_number, :]
            x_train_set = np.vstack((x_train, sample_test))
            y_train_set = np.vstack((y_train, y_label))

            set_sample_number = x_train_set.shape[0]
            p_value_tmp = np.zeros(set_sample_number)

            # calculate the p_value
            for set_sample in range(set_sample_number):
                x_train_set_tmp = x_train_set
                y_train_set_tmp = y_train_set

                x_sample = x_train_set_tmp[set_sample, :]
                y_sample = y_train_set_tmp[set_sample, :]

                x_train_set_tmp = np.delete(x_train_set_tmp, set_sample, axis=0)
                y_train_set_tmp = np.delete(y_train_set_tmp, set_sample, axis=0)

                same_label_index = (y_train_set_tmp == y_sample).ravel()
                diff_lable_index = (y_train_set_tmp != y_sample).ravel()

                same_label_sample = x_train_set_tmp[same_label_index]   # same label sample set
                diff_lable_sample = x_train_set_tmp[diff_lable_index]   # different label set

                dist_same_result = []
                dist_diff_result = []
                for num in range(same_label_sample.shape[0]):
                    dist_same = np.linalg.norm(same_label_sample[num, :] - x_sample)
                    if num == 0:
                        dist_same_result = [dist_same]
                    else:
                        dist_same_result = np.vstack((dist_same_result, dist_same))
                for num in range(diff_lable_sample.shape[0]):
                    dist_diff = np.linalg.norm(diff_lable_sample[num, :] - x_sample)
                    if num == 0:
                        dist_diff_result = [dist_diff]
                    else:
                        dist_diff_result = np.vstack((dist_diff_result, dist_diff))

                dist_same_result.sort(axis=0)
                dist_diff_result.sort(axis=0)
                same_value_simple = np.sum(dist_same_result[0:k])
                diff_value_simple = np.sum(dist_diff_result[0:k])
                p_value_tmp[set_sample] = same_value_simple / diff_value_simple


            p_value_numetrator = np.sum(p_value_tmp > p_value_tmp[-1])

            p_value_denominator = float(p_value_tmp.shape[0])
            predict[label_number] = p_value_numetrator / p_value_denominator
        print('the '+str(sample)+' is done')
        p_value[sample, :] = predict

    return p_value


def CP_KNN_offline(p_value):
    prediction_number = p_value.shape[0]
    p_value_prediction = np.zeros(prediction_number).reshape((prediction_number, 1))

    for col in range(prediction_number):
        p_value_prediction[col, 0] = p_value[col, :].argmax()
        print('the' + str(col) + ' prediction is : ' + str(p_value_prediction[col, 0]))

    return p_value_prediction


