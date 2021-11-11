import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)

    cols = list()
    # input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    # forecast sequence (t, t+1, ... t+n_out-1)
    for i in range(0, n_out):
        cols.append(df.shift(-i))

    # put it all together
    agg = pd.concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg.values

# split a univariatedataset into train/test sets


def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

# fit an xgboost modeland make a one step prediction


def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))

    return yhat[0]


# walk-forwardvalidation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the testset
    for i in range(len(test)):
        # split test row into input andoutput columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make aprediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list ofpredictions
        predictions.append(yhat)
        # add actual observation tohistory for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f,predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)

    return error, test[:, -1], predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast monthly births with xgboost")
    parser.add_argument('--data_filename', type=str, default="daily-total-female-births.csv",
                        help="Daily total female births dataset.")
    parser.add_argument('--n_in', type=int, default=3,
                        help="Input sequence length.")
    parser.add_argument('--n_out', type=int, default=1,
                        help="Predictive step size.")
    parser.add_argument('--dropnan', type=bool, default=False,
                        help="Whether to dropout the nan data.")
    parser.add_argument('--n_test', type=int, default=12,
                        help="Number of test data.")
    config = parser.parse_args()

    # load the dataset
    series = pd.read_csv(config.data_filename, header=0, index_col=0)
    values = series.values
    # print(series, values)

    # transform the timeseries data into supervised learning
    data = series_to_supervised(
        values, n_in=config.n_in, n_out=config.n_out, dropnan=config.dropnan)

    # evaluate
    mae, y, yhat = walk_forward_validation(data, config.n_test)
    print('MAE: %.3f' % mae)

    # plot expected vspreducted
    plt.plot(y, label='Expected')
    plt.plot(yhat, label='Predicted')
    plt.legend()
    plt.show()
