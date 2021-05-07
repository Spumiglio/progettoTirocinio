from multiprocessing import cpu_count
from warnings import catch_warnings, filterwarnings

import pandas as pd
from statistics import mean
from statistics import stdev
from math import sqrt

from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from forecasting import *


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
    errors = forecast_error(forecast_data, test_data) ** 2
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
        summ += abs(series[t] - series[t - lag])
    return summ


def prediction_interval(forecast_series, c=95):
    c_s = {80: 1.28, 85: 1.44, 90: 1.64, 95: 1.96, 96: 2.05, 97: 2.17, 98: 2.33, 99: 2.58}
    return forecast_series - (c_s[c] * stdev(forecast_series)), forecast_series + (c_s[c] * stdev(forecast_series))


def evaluate_simple_forecasts(df_train, df_test, data_column_name, season=25):
    # naive
    df_train_copy = df_train.copy()
    naive_errors = {}
    for i in range(0, len(df_test.index)):
        week_to_forecast = df_train_copy.index[df_train_copy.index.size - 1]
        forecast_date, forecast_value = naive(df_train_copy[data_column_name], week_to_forecast)
        df_train_copy.loc[forecast_date] = forecast_value
    naive_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    naive_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    naive_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    naive_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # seasonal naive
    df_train_copy = df_train.copy()
    seasonal_naive_errors = {}
    for i in range(0, len(df_test.index)):
        week_to_forecast = df_train_copy.index[df_train_copy.index.size - 1]
        forecast_date, forecast_value = seasonal_naive_forecasting(df_train_copy[data_column_name], week_to_forecast,
                                                                   season, 1)
        df_train_copy.loc[forecast_date] = forecast_value
    seasonal_naive_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    seasonal_naive_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    seasonal_naive_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    seasonal_naive_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # average
    df_train_copy = df_train.copy()
    average_errors = {}
    for i in range(0, len(df_test.index)):
        week_to_forecast = df_train_copy.index[df_train_copy.index.size - 1]
        forecast_date, forecast_value = average_forecasting(df_train_copy[data_column_name], week_to_forecast)
        df_train_copy.loc[forecast_date] = forecast_value
    average_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    average_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    average_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    average_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # drift
    df_train_copy = df_train.copy()
    drift_errors = {}
    for i in range(0, len(df_test.index)):
        df_train_copy = driftmethod(df_train_copy)
    drift_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    drift_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    drift_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    drift_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # Holt-Winters ETS
    df_train_copy = df_train.copy()
    holt_winter_errors = {}
    df_train_copy = seasonalExp_smoothing(df_train_copy, len(df_test.index))
    holt_winter_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    holt_winter_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    holt_winter_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    holt_winter_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # Simple ETS
    df_train_copy = df_train.copy()
    simple_ets_error = {}
    df_train_copy = smpExpSmoth(df_train_copy, len(df_test.index))
    simple_ets_error['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    simple_ets_error['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    simple_ets_error['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    simple_ets_error['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    errors = {'N': sum([naive_errors['MAE'], naive_errors['RMSE'], naive_errors['MAPE'], naive_errors['MASE']]),
              'SN': sum([seasonal_naive_errors['MAE'], seasonal_naive_errors['RMSE'], seasonal_naive_errors['MAPE'],
                         seasonal_naive_errors['MASE']]),
              'AVG': sum(
                  [average_errors['MAE'], average_errors['RMSE'], average_errors['MAPE'], average_errors['MASE']]),
              'DR': sum([drift_errors['MAE'], drift_errors['RMSE'], drift_errors['MAPE'], drift_errors['MASE']]),
              'HW': sum([holt_winter_errors['MAE'], holt_winter_errors['RMSE'], holt_winter_errors['MAPE'],
                         holt_winter_errors['MASE']]),
              'SES': sum([simple_ets_error['MAE'], simple_ets_error['RMSE'], simple_ets_error['MAPE'],
                          simple_ets_error['MASE']])}


    print(errors)
    keys = list(errors.keys())
    vals = list(errors.values())
    return keys[vals.index(min(vals))]


def evaluate_sarima_forecasts(df):
    # TODO
    pass


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    train, test = datasplitter(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        sarima = sarima_forecast_test(history, cfg)
        predictions.append(sarima)
        history.append(test[i])
    error = measure_rmse(test, predictions)
    # error = rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, n_test, cfg):
    result = None
    key = str(cfg)
    try:
        with catch_warnings():
            filterwarnings("ignore")
            result = walk_forward_validation(data, n_test, cfg)
    except:
        error = None
    # if result is not None:
    #     print(' > Model[%s] %.3f' % (key, result))
    return key, result


# grid search configs
def grid_search(data, cfg_list, n_test):
    scores = None
    executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
    tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
    scores = executor(tasks)
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models
