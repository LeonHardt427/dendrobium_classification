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
    prediction : DataFrame
        prediction with p_value

    Returns
    -------
    df_p: DataFrame
        the result of the max p_value
    """
    return prediction.max(axis=1)


def force_prediction(prediction):
    """
    Parameters
    ----------
    prediction : DataFrame
        prediction with p_value

    Returns
    -------
    df_p: DataFrame
        the result with the max p_value
    """
    return prediction.idxmax(axis=1)


if __name__ == "__main__":
    a = np.arange(12).reshape((4, 3))
    df = pd.DataFrame(a, columns=['a', 'b', 'c'])
    print(force_prediction_value(df))
    print(force_prediction(df))

