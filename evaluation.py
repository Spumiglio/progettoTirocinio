import itertools
from multiprocessing import cpu_count
from warnings import catch_warnings, filterwarnings

from statistics import mean
from statistics import stdev
from math import sqrt

from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

from dataPreparation import datasplitter
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


def evaluate_simple_forecasts(df_train, df_test, data_column_name, config, models, weight, forecast_driftict,
                              season=26):
    # Naive
    df_train_copy = df_train.copy()
    naive_errors = {}
    last_week = df_train_copy.index[df_train_copy.index.size - 1]
    naive(df_train_copy, last_week, week_to_forecast=len(df_train_copy.index))
    naive_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    naive_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    # naive_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    naive_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # Seasonal naive
    df_train_copy = df_train.copy()
    seasonal_naive_errors = {}
    last_week = df_train_copy.index[df_train_copy.index.size - 1]
    seasonal_naive_forecasting(df_train_copy, last_week, 26, 1, week_to_forecast=len(df_train_copy.index))
    seasonal_naive_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    seasonal_naive_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    # seasonal_naive_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    seasonal_naive_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # Average
    df_train_copy = df_train.copy()
    average_errors = {}
    last_week = df_train_copy.index[df_train_copy.index.size - 1]
    average_forecasting(df_train_copy, last_week, week_to_forecast=len(df_train_copy.index))
    average_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    average_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    # average_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    average_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # Drift
    df_train_copy = df_train.copy()
    drift_errors = {}
    last_week = df_train_copy.index[df_train_copy.index.size - 1]
    driftmethod(df_train_copy, last_week, week_to_forecast=len(df_train_copy.index))
    drift_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    drift_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    # drift_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    drift_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # Holt-Winters ETS
    df_train_copy = df_train.copy()
    holt_winter_errors = {}
    df_train_copy = seasonalExp_smoothing(df_train_copy, len(df_test.index))
    holt_winter_errors['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    holt_winter_errors['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    # holt_winter_errors['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    holt_winter_errors['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # Simple ETS
    df_train_copy = df_train.copy()
    simple_ets_error = {}
    df_train_copy = smpExpSmoth(df_train_copy, len(df_test.index))
    simple_ets_error['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    simple_ets_error['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    # simple_ets_error['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    simple_ets_error['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    # Aggregate Models
    aggregate_error = {}
    df_aggregate = aggregate_models(models)
    aggregate_error['MAE'] = mae(df_aggregate[data_column_name], df_test[data_column_name])
    aggregate_error['RMSE'] = rmse(df_aggregate[data_column_name], df_test[data_column_name])
    aggregate_error['MASE'] = mase(df_aggregate[data_column_name], df_test[data_column_name])

    # Aggregate Weighted Models
    aggregate_weighted_error = {}
    df_aggregate_weighted = aggregate_weighted(weight, forecast_driftict, len(df_test.index))
    aggregate_weighted_error['MAE'] = mae(df_aggregate_weighted[data_column_name], df_test[data_column_name])
    aggregate_weighted_error['RMSE'] = rmse(df_aggregate_weighted[data_column_name], df_test[data_column_name])
    aggregate_weighted_error['MASE'] = mase(df_aggregate_weighted[data_column_name], df_test[data_column_name])

    # Sarima
    df_train_copy = df_train.copy()
    sarima_ets_error = {}
    df_train_copy = sarima_forecast(df_train_copy, config, len(df_test.index))
    sarima_ets_error['MAE'] = mae(df_train_copy[data_column_name], df_test[data_column_name])
    sarima_ets_error['RMSE'] = rmse(df_train_copy[data_column_name], df_test[data_column_name])
    # sarima_ets_error['MAPE'] = mape(df_train_copy[data_column_name], df_test[data_column_name])
    sarima_ets_error['MASE'] = mase(df_train_copy[data_column_name], df_test[data_column_name])

    errors = {'N': [naive_errors['MAE'], naive_errors['RMSE'], naive_errors['MASE'],
                    sum([naive_errors['MAE'], naive_errors['RMSE'], naive_errors['MASE']])],
              'SN': [seasonal_naive_errors['MAE'], seasonal_naive_errors['RMSE'],
                     seasonal_naive_errors['MASE'], sum([seasonal_naive_errors['MAE'], seasonal_naive_errors['RMSE'],
                                                         seasonal_naive_errors['MASE']])],
              'AVG': [average_errors['MAE'], average_errors['RMSE'], average_errors['MASE'], sum(
                  [average_errors['MAE'], average_errors['RMSE'], average_errors['MASE']])],
              'DR': [drift_errors['MAE'], drift_errors['RMSE'], drift_errors['MASE'],
                     sum([drift_errors['MAE'], drift_errors['RMSE'], drift_errors['MASE']])],
              'HW': [holt_winter_errors['MAE'], holt_winter_errors['RMSE'],
                     holt_winter_errors['MASE'], sum([holt_winter_errors['MAE'], holt_winter_errors['RMSE'],
                                                      holt_winter_errors['MASE']])],
              'SES': [simple_ets_error['MAE'], simple_ets_error['RMSE'],
                      simple_ets_error['MASE'], sum([simple_ets_error['MAE'], simple_ets_error['RMSE'],
                                                     simple_ets_error['MASE']])],
              'SRM': [sarima_ets_error['MAE'], sarima_ets_error['RMSE'],
                      sarima_ets_error['MASE'], sum([sarima_ets_error['MAE'], sarima_ets_error['RMSE'],
                                                     sarima_ets_error['MASE']])],
              'AGG': [aggregate_error['MAE'], aggregate_error['RMSE'],
                      aggregate_error['MASE'], sum([aggregate_error['MAE'], aggregate_error['RMSE'],
                                                    aggregate_error['MASE']])],
              'AGG-WGT': [aggregate_weighted_error['MAE'], aggregate_weighted_error['RMSE'],
                          aggregate_weighted_error['MASE'],
                          sum([aggregate_weighted_error['MAE'], aggregate_weighted_error['RMSE'],
                               aggregate_weighted_error['MASE']])],
              }
    # print(errors)
    return errors


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
    p_params = [0, 1, 2, 3, 4, 5]
    d_params = [0, 1]
    q_params = [0, 1, 2, 3, 4, 5]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2, 3, 4, 5]
    D_params = [0, 1]
    Q_params = [0, 1, 2, 3, 4, 5]
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


def best_aggregate_config(models, test):
    error_dict = {}
    all_combinations = []
    key_list = models.keys()

    for r in range(len(key_list) + 1):
        combinations_object = itertools.combinations(key_list, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list
    del all_combinations[:8]

    for combination in all_combinations:
        comb = list(combination)
        df_list = [models[x] for x in comb]
        df = aggregate_models(df_list)
        error = mae(df['vendite'], test['vendite'])
        error_dict[combination] = error

    keys = list(error_dict.keys())
    vals = list(error_dict.values())
    return [keys[vals.index(min(vals))], error_dict]


def model_weighted(models, test):
    error_dict = {}
    for model in models:
        error = mae(models[model]['vendite'], test['vendite'])
        error_dict[model] = error
    alpha = {k: v for k, v in sorted(error_dict.items(), key=lambda item: item[1])}
    keys = list(alpha.keys())
    alpha[keys[0]] = 0.4
    alpha[keys[1]] = 0.25
    alpha[keys[2]] = 0.16
    alpha[keys[3]] = 0.1
    alpha[keys[4]] = 0.05
    alpha[keys[5]] = 0.03
    alpha[keys[6]] = 0.01

    return alpha
