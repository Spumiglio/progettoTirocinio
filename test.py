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

    # dativendita_nero = filter_by_color(dativendita, 'fantasia')

    train, test = datasplitter(dativendita)

    dativendita_colore = weeksdistrubution(train)

    dativendita_colore_test = weeksdistrubution(test)
    forecast_index = dativendita_colore.index.size - 1

    print('Best method: ' + evaluate_simple_forecasts(dativendita_colore, dativendita_colore_test, 'vendite', season=26))

    # Average
    for i in range(0, 27):
        week_to_forecast = dativendita_colore.index[dativendita_colore.index.size - 1]
        forecast_date, forecast_value = average_forecasting(dativendita_colore['vendite'], week_to_forecast)
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(dativendita_colore)
    # plot_dataframe(dativendita_colore,dativendita_colore_test, plot_name="Average", forecasting_indexes=forecast_index)
    print("Average MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))


    # Seasonal Naive
    dativendita_colore = weeksdistrubution(train)
    # testing seasonal naive
    for i in range(0, 27):
        week_to_forecast = dativendita_colore.index[dativendita_colore.index.size - 1]
        forecast_date, forecast_value = seasonal_naive_forecasting(dativendita_colore['vendite'], week_to_forecast, 26, 1)
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)
    print("Seasonal Naive MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    #  Naive
    dativendita_colore = weeksdistrubution(train)
    for i in range(0, 27):
        forecast_date, forecast_value = naive(dativendita_colore,dativendita_colore.index[dativendita_colore.index.size - 1])
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(dativendita_colore,dativendita_colore_test, plot_name="Naive", forecasting_indexes=forecast_index)
    print("Naive MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    # Drift
    dativendita_colore = weeksdistrubution(train)
    for i in range(0, 27):
        driftmethod(dativendita_colore)
    # plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name="Drift", forecasting_indexes=forecast_index)
    print("Drift MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    # Seasonal Exp Smoothing
    dativendita_colore = weeksdistrubution(train)
    seasonalExp_smoothing(dativendita_colore,27)
    # plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name="HoltWinter", forecasting_indexes=forecast_index)
    print("ETS MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    # Simple Exp Smoothing
    dativendita_colore = weeksdistrubution(train)
    smpExpSmoth(dativendita_colore, 27)
    # plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name='simpleExpSmothing', forecasting_indexes=forecast_index)
    print("Simple ETS MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    # Sarima

    #     test score model
    d = dativendita_colore['vendite'].values.tolist()
    n_test = 10
    # model configs
    cfg_list = sarima_configs()
    # grid search
    scores = grid_search(d, cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)

    cfg = ast.literal_eval(cfg)
    dativendita_colore = weeksdistrubution(train)
    sarima_forecast(dativendita_colore, cfg, 27)
    plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name='Arima', forecasting_indexes=forecast_index)

if __name__ == '__main__':
    main()
