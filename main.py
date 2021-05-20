from forecasting import *
from plotter import *
from evaluation import *
import pandas as pd
import ast
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from scipy import stats

def main():
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    dativendita = best20color(dativendita)
    datetoweek(dativendita)

    dativendita = filter_by_color(dativendita, 'nero')
    dativendita_colore = weeksdistrubution(dativendita)

    dativendita_colore = fill_missing_data(dativendita_colore, start=dativendita_colore.index[0],
                                           end=dativendita_colore.index[len(dativendita_colore.index) - 1],
                                           fill_mode='V', fill_value=0)

    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))

    forecast_index = train.index.size - 1

    # Autocorrelazione con pandas
    autocorrelazione = train.squeeze().autocorr()

    # Autocorrelazione con Statsmodel
    plot_acf(train, title="autocorrelazione train", lags=None)
    pyplot.show()

    last_week = train.index[train.index.size - 1]

    # Rimozione Outliers
    train = train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]
    test = test[(np.abs(stats.zscore(test)) < 3).all(axis=1)]

    # Average
    df_avg = average_forecasting(train.copy(), last_week, week_to_forecast=len(test.index))
    plot_dataframe(df_avg, test, plot_name="Average", forecasting_indexes=forecast_index)

    # Seasonal Naive
    df_sn = seasonal_naive_forecasting(train.copy(), last_week, 26, 1, week_to_forecast=len(test.index))
    plot_dataframe(df_sn, test, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)

    #  Naive
    df_n = naive(train.copy(), last_week, week_to_forecast=len(test.index))
    plot_dataframe(df_n, test, plot_name="Naive", forecasting_indexes=forecast_index)

    # Drift
    df_d = driftmethod(train.copy(), last_week, week_to_forecast=len(test.index))
    plot_dataframe(df_d, test, plot_name="Drift", forecasting_indexes=forecast_index)

    # Seasonal Exp Smoothing
    df_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False)
    plot_dataframe(df_hw, test, plot_name="Holt-Winters", forecasting_indexes=forecast_index)

    # Simple Exp Smoothing
    df_ses = smpExpSmoth(train.copy(), len(test.index))
    plot_dataframe(df_ses, test, plot_name='Simple Exponential Smoothing', forecasting_indexes=forecast_index)

    # Sarima
    # test score model
    # Score Model Sarima
    scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3) #10
    # List top 3 configs
    print('Top 3:')
    for cfg, error in scores[:3]:
        print(cfg, error)

    cfg = ast.literal_eval(cfg)
    df_sar = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=False)
    plot_dataframe(df_sar, test, plot_name='Arima', forecasting_indexes=forecast_index)

    # Aggregate Testing
    models = [df_hw, df_d]
    aggregate = aggregate_models(models)
    plot_dataframe(aggregate, test, plot_name="Aggregate", forecasting_indexes=forecast_index)

    print('Best method: ' + evaluate_simple_forecasts(train, test, 'vendite', cfg, models))



if __name__ == '__main__':
    main()
