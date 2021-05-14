import pandas as pd
from dateutil import parser
from statsmodels.tsa.statespace.sarimax import SARIMAX

from dataPreparation import *
from statsmodels.tsa.holtwinters import *


def naive(df,week, week_to_forecast=27):
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


def seasonalExp_smoothing(df, week_to_predict=1):
    dateiso = []
    for week in df.index:
        dateiso.append(dateutil.parser.isoparse(week))
    dateiso = pd.DatetimeIndex(dateiso).to_period('W')
    series = pd.Series(data=df['vendite'].values, index=dateiso)
    model = ExponentialSmoothing(series, seasonal_periods=26, seasonal='add', initialization_method="estimated").fit()
    predict = model.forecast(week_to_predict)
    week = df.index[df.index.size - 1]
    for i in range(0, week_to_predict):
        week = add_week(week, 1)
        df.loc[week] = predict.iloc[i]
    return df


def sarima_forecast_test(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


def sarima_forecast(df, config, weektopredict=1):
    dateiso = []
    for week in df.index:
        dateiso.append(dateutil.parser.isoparse(week))
    dateiso = pd.DatetimeIndex(dateiso).to_period('W')
    series = pd.Series(data=df['vendite'].values, index=dateiso)
    config = list(config)
    order = config[0]
    sorder = config[1]
    trend = config[2]

    model = SARIMAX(series, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    model_fit = model.fit(disp=False)
    predict = model_fit.get_prediction()
    week = df.index[df.index.size - 1]
    for i in range(0, weektopredict):
        week = add_week(week, 1)
        df.loc[week] = predict.predicted_mean.iloc[i]
    return df


def aggregate_models(models):
    df = sum(models)/len(models)
    return df
