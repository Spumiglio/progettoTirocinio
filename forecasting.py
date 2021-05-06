import pandas as pd
from dateutil import parser
from dataPreparation import *
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def smpExpSmoth(df, num_of_forcast):
    dateiso = []
    for week in df.index:
        dateiso.append(dateutil.parser.isoparse(week))
    dateiso = pd.DatetimeIndex(dateiso).to_period('W')
    newse = pd.Series(data=df['vendite'].values, index=dateiso)
    model = SimpleExpSmoothing(newse, initialization_method='estimated').fit(smoothing_level=0.6, optimized=False)
    predict = model.forecast(num_of_forcast)
    week = df.index[df.index.size - 1]
    for i in range(0, num_of_forcast):
        week = add_week(week, 1)
        df.loc[week] = predict.iloc[i]
    return df


def naive(series_to_forecast, week_to_forecast):
    return add_week(week_to_forecast, 1), series_to_forecast.tail(1).values[0]


def driftmethod(df):
    y_t = df.loc[df.index[len(df) - 1]]['vendite']
    m = (y_t - df.loc[df.index[0]]['vendite']) / len(df)
    h = 1
    valforecast = y_t + m * h
    new_week = add_week(df.index[len(df) - 1], 1)
    df.loc[new_week] = valforecast
    return df


def average_forecasting(series_to_forecast, week_to_forecast):
    avg = int(series_to_forecast.mean())
    return add_week(week_to_forecast, 1), avg


def seasonal_naive_forecasting(series_to_forecast, week_to_forecast, season_length, h):
    k = int((h - 1) / season_length)
    return add_week(week_to_forecast, 1), series_to_forecast[series_to_forecast.size + h - season_length * (k + 1)]

def seasonalExp_smoothing(df,weektopredict=1):
    dateiso = []
    for week in df.index:
        dateiso.append(dateutil.parser.isoparse(week))
    dateiso = pd.DatetimeIndex(dateiso).to_period('W')
    series = pd.Series(data=df['vendite'].values, index=dateiso)
    model = ExponentialSmoothing(series, seasonal_periods=26, seasonal='add', use_boxcox=False, initialization_method="estimated").fit()
    predict = model.forecast(weektopredict)
    week = df.index[df.index.size - 1]
    for i in range(0, weektopredict):
        week = add_week(week, 1)
        df.loc[week] = predict.iloc[i]
    return df