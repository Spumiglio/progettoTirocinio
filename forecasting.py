from dataPreparation import*

def naive(series_to_forecast, week_to_forecast):
    return  add_week(week_to_forecast,1), series_to_forecast.tail(1).values[0]


def driftmethod(df):
    y_t = df.loc[df.index[len(df) - 1]]['vendite']
    m = (y_t - df.loc[df.index[0]]['vendite']) / len(df)
    h = 1
    valforecast = y_t + m * h
    new_week= add_week(df.index[len(df) - 1],1)
    df.loc[new_week]=valforecast
    return df


def average_forecasting(series_to_forecast, week_to_forecast):
    avg = int(series_to_forecast.mean())
    return add_week(week_to_forecast, 1), avg


def seasonal_naive_forecasting(series_to_forecast, week_to_forecast, season_length, h):
    k = int((h - 1) / season_length)
    return add_week(week_to_forecast, 1), series_to_forecast[series_to_forecast.size + h - season_length*(k+1)]

