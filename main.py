from forecasting import *
from plotter import *
from evaluation import *
import pandas as pd
import ast


def main():
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    dativendita = best20color(dativendita)
    datetoweek(dativendita)

    dativendita = filter_by_color(dativendita, 'panna')
    dativendita_colore = weeksdistrubution(dativendita)

    dativendita_colore = fill_missing_data(dativendita_colore, start=dativendita_colore.index[0],
                                           end=dativendita_colore.index[len(dativendita_colore.index) - 1],
                                           fill_mode='V', fill_value=0)

    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))

    forecast_index = train.index.size - 1


    # Average
    last_week = train.index[train.index.size - 1]
    average_forecasting(train, last_week, week_to_forecast=len(test.index))
    # plot_dataframe(train, test, plot_name="Average", forecasting_indexes=forecast_index)

    # Seasonal Naive
    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))
    last_week = train.index[train.index.size - 1]
    seasonal_naive_forecasting(train, last_week, 26, 1, week_to_forecast=len(test.index))
    # plot_dataframe(train, test, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)

    #  Naive
    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))
    last_week = train.index[train.index.size - 1]
    naive(train, last_week, week_to_forecast=len(test.index))
    # plot_dataframe(train, test, plot_name="Naive", forecasting_indexes=forecast_index)

    # Drift
    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))
    last_week = train.index[train.index.size - 1]
    driftmethod(train, last_week, week_to_forecast=len(test.index))
    # plot_dataframe(train, test, plot_name="Drift", forecasting_indexes=forecast_index)

    # Seasonal Exp Smoothing
    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))
    seasonalExp_smoothing(train, len(test.index))
    # plot_dataframe(train, test, plot_name="Holt-Winters", forecasting_indexes=forecast_index)

    # Simple Exp Smoothing
    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))
    smpExpSmoth(train, len(test.index))
    # plot_dataframe(train, test, plot_name='Simple Exponential Smoothing', forecasting_indexes=forecast_index)

    # Sarima

    #     test score model
    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))
    # Score Model Sarima
    scores = grid_search(train['vendite'].values.tolist(), sarima_configs(), n_test=3)
    # List top 3 configs
    print('Top 3:')
    for cfg, error in scores[:3]:
        print(cfg, error)

    cfg = ast.literal_eval(cfg)
    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))
    sarima_forecast(train, cfg, 27)
    plot_dataframe(train, test, plot_name='Arima', forecasting_indexes=forecast_index)

    train, test = data_splitter(dativendita_colore, int(len(dativendita_colore.index) * 0.2))
    print('Best method: ' + evaluate_simple_forecasts(train, test, 'vendite', cfg))


if __name__ == '__main__':
    main()
