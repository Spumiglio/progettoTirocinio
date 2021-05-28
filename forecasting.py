import dateutil
import pandas as pd
from statsmodels.tsa._stl import STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.holtwinters import *

from dataPreparation import add_week, box_cox_transformation, remove_outliers


def naive(df, week, week_to_forecast=27):
    for i in range(0, week_to_forecast):
        week = add_week(week, 1)
        df.loc[week] = df.tail(1).values[0]
    return df


def driftmethod(df, week, week_to_forecast=27):
    for i in range(0, week_to_forecast):
        y_t = df.loc[df.index[len(df) - 1]]['vendite']
        m = (y_t - df.loc[df.index[0]]['vendite']) / len(df)
        h = 1
        val_forecast = y_t + m * h
        week = add_week(week, 1)
        df.loc[week] = val_forecast
    return df


def average_forecasting(df, last_week, week_to_forecast=27):
    for i in range(0, week_to_forecast):
        last_week = add_week(last_week, 1)
        df.loc[last_week] = int(df.mean())
    return df


def seasonal_naive_forecasting(df, last_week, season_length, h, week_to_forecast=27):
    k = int((h - 1) / season_length)
    for i in range(0, week_to_forecast):
        last_week = add_week(last_week, 1)
        df.loc[last_week] = df.iloc[df.size + h - season_length * (k + 1)]
    return df


def smpExpSmoth(df, num_of_forecast):
    dateiso = []
    for week in df.index:
        dateiso.append(dateutil.parser.isoparse(week))
    dateiso = pd.DatetimeIndex(dateiso).to_period('W')
    series = pd.Series(data=df['vendite'].values, index=dateiso)
    model = SimpleExpSmoothing(series, initialization_method='estimated').fit(smoothing_level=0.6, optimized=True)
    predict = model.forecast(num_of_forecast)
    week = df.index[df.index.size - 1]
    for i in range(0, num_of_forecast):
        week = add_week(week, 1)
        df.loc[week] = predict.iloc[i]
    return df


def seasonalExp_smoothing(df, week_to_predict=1, decompositon=False, box_cox=0, rmv_outliers=True):
    dateiso = []
    for week in df.index:
        dateiso.append(dateutil.parser.isoparse(week))
    dateiso = pd.DatetimeIndex(dateiso).to_period('W')

    df_o = df.copy()
    if rmv_outliers:
        df_o = remove_outliers(df)
    if box_cox != 0:
        df_o = box_cox_transformation(df_o.copy(), box_cox)
    series = pd.Series(data=df_o['vendite'].values, index=dateiso)

    if decompositon:
        model = STLForecast(series, ExponentialSmoothing, period=26,
                            model_kwargs=dict(seasonal_periods=26, seasonal='add', initialization_method="estimated"))
        model_fitted = model.fit()
    else:
        model = ExponentialSmoothing(series, seasonal_periods=26, seasonal='add',
                                     initialization_method='estimated')
        model_fitted = model.fit()
    if box_cox != 0:
        predict = box_cox_transformation(model_fitted.forecast(week_to_predict), box_cox, reverse=True)
    else:
        predict = model_fitted.forecast(week_to_predict)
    week = df.index[df.index.size - 1]
    for i in range(0, week_to_predict):
        week = add_week(week, 1)
        df.loc[week] = predict.iloc[i]
    return df


def sarima_forecast_test(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(history), len(history))  # TODO get_prediction() non va bene!
    return yhat[0]


def sarima_forecast(df, config, weektopredict=1, decomposition=False, box_cox=0, rmv_outliers=True):
    dateiso = []
    for week in df.index:
        dateiso.append(dateutil.parser.isoparse(week))
    dateiso = pd.DatetimeIndex(dateiso).to_period('W')

    df_o = df.copy()
    if rmv_outliers:
        df_o = remove_outliers(df)
    if box_cox != 0:
        df_o = box_cox_transformation(df_o.copy(), box_cox)
    series = pd.Series(data=df_o['vendite'].values, index=dateiso)
    config = list(config)
    order = config[0]
    sorder = config[1]
    trend = config[2]

    if decomposition:
        model = STLForecast(series, SARIMAX, period=26,
                            model_kwargs=dict(order=order, seasonal_order=sorder, trend=trend))
        model_fit = model.fit(fit_kwargs=dict(disp=False))
    else:
        model = SARIMAX(series, order=order, seasonal_order=sorder, trend=trend)
        model_fit = model.fit(disp=False)
    if box_cox != 0:
        predict = box_cox_transformation(model_fit.forecast(weektopredict), box_cox, reverse=True)
    else:
        predict = model_fit.forecast(weektopredict)
    week = df.index[df.index.size - 1]
    for i in range(0, weektopredict):
        week = add_week(week, 1)
        df.loc[week] = predict.iloc[i]
    return df


# method puo' essere "STL" per una decomposizione di Loess o "CL" per una decomposizione classica
def decompose(df, method='STL'):
    if method == 'STL':
        stl_object = STL(df, period=26, seasonal=7, robust=True, seasonal_deg=1)
        stl_fitted = stl_object.fit()
        stl_fitted.plot()
        return stl_fitted.trend, stl_fitted.seasonal, stl_fitted.resid
    elif method == 'CL':
        stl_object = seasonal_decompose(df, period=26)
        stl_object.plot()
        return stl_object.trend, stl_object.seasonal, stl_object.resid


def aggregate_models(models):
    df = sum(models)/len(models)
    return df


def aggregate_weighted(weights, modelsdict, forecast_size):
    df_copy = list(modelsdict.values())[0].copy()
    somma = 0
    for mod in weights:
        somma += modelsdict[mod].tail(forecast_size)*weights[mod]
    for index, row in somma.iterrows():
        df_copy.loc[index] = row["vendite"]
    return df_copy
