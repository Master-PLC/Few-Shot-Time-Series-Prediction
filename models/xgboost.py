#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :xgboost.py
@Description  :
@Date         :2021/11/08 15:07:19
@Author       :Arctic Little Pig
@Version      :1.0
'''

import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from numpy.lib.npyio import save
from utils.metrics import get_metrics

from xgboost import XGBRegressor


class XGBoost(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_in = config.n_in
        self.n_out = config.n_out
        self.estimators = config.estimators
        self.self_training = True
        self.train_type_list = ["walk_forward_validation", "default"]
        self.training_type = self.train_type_list[0]

        self.save_path = os.path.join(config.checkpoint, config.model)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        saved_model = os.listdir(self.save_path)
        # print(saved_model)
        if saved_model:
            saved_model = [os.path.join(self.save_path, fn)
                           for fn in saved_model]
            saved_model.sort(key=lambda fn: os.path.getmtime(fn))
            newest_model = saved_model[-1]
            self.regressor = joblib.load(saved_model[-1])
            print(f"successfully loaded the newest model: {newest_model}.")
        else:
            self.regressor = XGBRegressor(
                objective='reg:squarederror', n_estimators=self.estimators)

    def forward(self, X):
        batch_size = len(X.shape)
        if batch_size == 1:
            X = [X]

        y = self.regressor.predict(np.asarray(X))

        if batch_size == 1:
            return y[0]
        else:
            return y

    def train(self, train, test):
        if self.training_type == "walk_forward_validation":
            predictions = list()
            # seed history with training dataset
            history = [x for x in train]

            # step over each time-step in the testset
            for i in range(len(test)):
                # split test row into input andoutput columns
                testX, testy = test[i, :-self.n_out], test[i, -self.n_out:]

                # fit model on history and make aprediction
                # transform train data list into array
                train = np.asarray(history)
                # split into input and output columns
                trainX, trainy = train[:, :-self.n_out], train[:, -self.n_out:]
                self.regressor.fit(trainX, trainy)
                yhat = self.forward(testX)
                # store forecast in list ofpredictions
                predictions.append(yhat)
                # add actual observation tohistory for the next loop
                del history[0]
                history.append(test[i])
                # summarize progress
                print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
        elif self.training_type == "default":
            trainX, trainy = train[:, :-self.n_out], train[:, -self.n_out:]
            self.regressor.fit(trainX, trainy)

            testX = test[:, :-self.n_out]
            predictions = self.forward(testX)
            # print(predictions)

        # estimate prediction error
        print("Training metrics: \n{}".format(
            get_metrics(np.array(predictions), test[:, -self.n_out:])))

        # plot expected vspreducted
        plt.plot(test[:, -self.n_out], ls=':', lw=3, label='Expected')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        plt.show()

        local_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        save_name = f"in{self.config.n_in}_out{self.config.n_out}_est{self.estimators}_{local_time}.pkl"
        joblib.dump(self.regressor, os.path.join(self.save_path, save_name))
