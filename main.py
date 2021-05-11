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

    dativendita = filter_by_color(dativendita, 'jeans')

    train, test = datasplitter(dativendita)

    dativendita_colore = weeksdistrubution(train)
    dativendita_colore_test = weeksdistrubution(test)

    fill_missing_data(dativendita_colore, start=dativendita_colore.index[0],
                      end=dativendita_colore.index[len(dativendita_colore.index) - 1], fill_mode='A')
    fill_missing_data(dativendita_colore_test, start=dativendita_colore_test.index[0],
                      end=dativendita_colore_test.index[len(dativendita_colore_test.index) - 1], fill_mode='A')

    forecast_index = dativendita_colore.index.size - 1



    # Average
    dativendita_colore = weeksdistrubution(train)
    last_week = dativendita_colore.index[dativendita_colore.index.size - 1]
    average_forecasting(dativendita_colore, last_week, week_to_forecast=len(dativendita_colore_test.index) )
    # plot_dataframe(dativendita_colore,dativendita_colore_test, plot_name="Average", forecasting_indexes=forecast_index)


    # Seasonal Naive
    dativendita_colore = weeksdistrubution(train)
    last_week = dativendita_colore.index[dativendita_colore.index.size - 1]
    seasonal_naive_forecasting(dativendita_colore, last_week, 26, 1, week_to_forecast=len(dativendita_colore_test.index) )
    # plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)

    #  Naive
    dativendita_colore = weeksdistrubution(train)
    last_week = dativendita_colore.index[dativendita_colore.index.size - 1]
    naive(dativendita_colore,last_week,week_to_forecast=len(dativendita_colore_test.index))
    # plot_dataframe(dativendita_colore,dativendita_colore_test, plot_name="Naive", forecasting_indexes=forecast_index)

    # Drift
    dativendita_colore = weeksdistrubution(train)
    last_week = dativendita_colore.index[dativendita_colore.index.size - 1]
    driftmethod(dativendita_colore, last_week,week_to_forecast=len(dativendita_colore_test.index))
    # plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name="Drift", forecasting_indexes=forecast_index)

    # Seasonal Exp Smoothing
    dativendita_colore = weeksdistrubution(train)
    seasonalExp_smoothing(dativendita_colore, len(dativendita_colore_test.index))
    # plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name="HoltWinter", forecasting_indexes=forecast_index)

    # Simple Exp Smoothing
    dativendita_colore = weeksdistrubution(train)
    smpExpSmoth(dativendita_colore, len(dativendita_colore_test.index))
    # plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name='simpleExpSmothing', forecasting_indexes=forecast_index)

    # Sarima

    #     test score model
    d = dativendita_colore['vendite'].values.tolist()
    n_test = 27
    cfg_list = sarima_configs()
    scores = grid_search(d, cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)

    cfg = ast.literal_eval(cfg)
    dativendita_colore = weeksdistrubution(train)
    sarima_forecast(dativendita_colore, cfg, 27)
    plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name='Arima', forecasting_indexes=forecast_index)


    dativendita_colore = weeksdistrubution(train)
    dativendita_colore_test = weeksdistrubution(test)
    print('Best method: ' + evaluate_simple_forecasts(dativendita_colore, dativendita_colore_test, 'vendite', cfg))
if __name__ == '__main__':
    main()
