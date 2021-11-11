#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :preprocess.py
@Description  :
@Date         :2021/11/08 10:31:17
@Author       :Arctic Little Pig
@Version      :1.0
'''

import pandas as pd


def read_data150(fp):
    series = pd.read_excel(fp, header=0, index_col=0)
    series = series.dropna(axis=1, how="all")

    values = series.values

    return values

# transform a time series dataset into a supervised learning dataset


def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    n_vars = 1 if type(data) is list else data.shape[1]

    X = data[:, :-1]
    Y = data[:, -1]
    df_X = pd.DataFrame(X)
    df_X = (df_X - df_X.min()) / (df_X.max() - df_X.min())
    df_Y = pd.DataFrame(Y)

    cols = list()
    # input sequence (t-n_in+1, ... t)
    for i in range(n_in, 0, -1):
        cols.append(df_X.shift(i-1))

    # forecast sequence (t, t+1, ... t+n_out-1)
    for i in range(0, n_out):
        cols.append(df_Y.shift(-i))

    # put it all together
    agg = pd.concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg.values


def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]
