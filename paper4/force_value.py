# -*- coding: utf-8 -*-
# @Time    : 2017/12/31 16:09
# @Author  : LeonHardt
# @File    : force_value.py


import numpy as np
import pandas as pd


def force_prediction_value(prediction):
    """

    Parameters
    ----------
    prediction : []
        prediction with p_value

    Returns
    -------
    result: [].(-1, 1)
        the result of the max p_value
    """
    idx_max = prediction.argmax(axis=1)
    result = np.zeros((prediction.shape[0]))
    for i in range(prediction.shape[0]):
        result[i] = prediction[i, idx_max[i]]
    result = result.reshape((-1, 1))
    return result


def force_prediction_correct(prediction, y):
    """
    Parameters
    ----------
    prediction : []
        prediction with p_value
    y: []
        the labels
    Returns
    -------
    correct: int
        the number of right prediction
    """
    correct = []
    max_prediction = force_prediction_value(prediction)
    for i in range(prediction.shape[0]):
        if i == 0:
            correct = prediction[i, :] == max_prediction[i]
        else:
            correct = np.vstack((correct, prediction[i, :] == max_prediction[i]))
    labels, y = np.unique(y, return_inverse=True)
    result = np.zeros((y.size,), dtype=bool)
    for i, y_ in enumerate(y):
        result[i] = correct[i, int(y_)]
    return np.sum(result)


def force_mean_errors(prediction, y):
    """Calculates the average error rate of a conformal classification model.
    """
    return 1 - (force_prediction_correct(prediction, y) / y.size)


if __name__ == "__main__":
    a = np.array([[1, 2, 3], [5, 7, 2], [3, 1, 2]])
    y = [2, 0, 1]
    print(force_prediction_value(a))
    print(force_prediction_correct(a, y))


