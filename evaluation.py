import pandas as pd
from statistics import mean
from math import sqrt


def forecast_error(forecast_data, test_data):
    errors = pd.Series()
    errors = errors.reindex(test_data.index)
    for i in test_data.index:
        error = test_data[i] - forecast_data[i]
        errors[i] = error
    return errors


# Mean Absolute Error
def mae(forecast_data, test_data):
    return mean(abs(forecast_error(forecast_data, test_data)))


# Root Mean Square Error
def rmse(forecast_data, test_data):
    errors = forecast_error(forecast_data, test_data)**2
    m = mean(errors)
    return sqrt(m)


# Mean Absolute Percentage Error
def mape(forecast_data, test_data):
    errors = forecast_error(forecast_data, test_data)
    percentage = 100 * (errors / test_data)
    return mean(abs(percentage))


# Mean Absolute Scaled Error
def mase(forecast_data, test_data, season=None):
    errors = forecast_error(forecast_data, test_data)
    if season is not None:
        q = (errors / ((1 / len(errors) - 1) * diff_val(test_data, season)))
    else:
        q = (errors / ((1 / len(errors) - 1) * diff_val(test_data)))

    return mean(abs(q))


# fa y(t) - y(t-1)
def diff_val(series, lag=1):
    summ = 0
    for t in range(lag, len(series)):
        summ += abs(series[t] - series[t-lag])
    return summ
