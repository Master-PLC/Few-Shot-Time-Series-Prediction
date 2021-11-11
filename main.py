#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :main.py
@Description  :
@Date         :2021/11/08 10:26:32
@Author       :Arctic Little Pig
@Version      :1.0
'''

import argparse
from os import name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from utils.metrics import get_metrics
from utils.name_to_model import name2model
from utils.preprocess import read_data150, series_to_supervised, train_test_split
from utils.seed_init import init_seed


def update_config():
    model_type = config.model.lower()
    if model_type == "xgboost":
        config.estimators = 1000  # Number of estimators
    elif model_type == "bht_arima":
        config.p = 3  # p-order
        config.d = 2  # d-order
        config.q = 1  # q-order
        config.taus = [1, 50]  # MDT-rank
        config.Rs = [50, 50]  # tucker decomposition ranks
        config.K = 3  # iterations
        config.tol = 0.001  # stop criterion
        config.Us_mode = 4  # orthogonality mode
        config.verbose = 0
        config.convergence_loss = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast monthly births with xgboost")
    parser.add_argument('--data_filename', type=str, default="./data/data150.xls",
                        help="Daily total female births dataset.")
    parser.add_argument('--n_in', type=int, default=1,
                        help="Input sequence length.")
    parser.add_argument('--n_out', type=int, default=1,
                        help="Predictive step size.")
    parser.add_argument('--dropnan', type=bool, default=False,
                        help="Whether to dropout the nan data.")
    parser.add_argument('--n_test', type=int, default=50,
                        help="Number of test data.")
    parser.add_argument('--init_seed', type=bool, default=True,
                        help="Whether to initialize the random seed.")
    parser.add_argument('--seed', type=int, default=2560,
                        help="Random seed.")
    parser.add_argument('--model', type=str, default="xgboost",
                        help="Name of model used for prediction.")
    parser.add_argument('--train', type=bool, default=True,
                        help="Whether to train the model.")
    parser.add_argument('--checkpoint', type=str, default="checkpoint",
                        help="Saved model directory.")
    config = parser.parse_args()

    if config.init_seed:
        init_seed(config.seed)

    # load the dataset
    values = read_data150(config.data_filename)
    # print(values)

    # transform the timeseries data into supervised learning
    data = series_to_supervised(
        values, n_in=config.n_in, n_out=config.n_out, dropnan=config.dropnan)
    # print(data)
    train, test = train_test_split(data, config.n_test)

    update_config()
    # print(config)

    model = name2model(config.model.lower())
    model = model(config)
    if config.train:
        if model.self_training:
            model.train(train, test)

    testX, testy = test[:, :-config.n_out], test[:, -config.n_out:]
    predictions = model(testX)
    print("Evaluation metrics: \n{}".format(get_metrics(predictions, testy)))
    # plot expected vspreducted
    y = np.concatenate((train[:, -config.n_out:], testy), axis=0)
    plt.plot(y, ls=':', lw=3, label='Expected')
    plt.plot(np.arange(100, 150), predictions, label='Predicted')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(testy, ls=':', lw=3, label='Expected')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()
