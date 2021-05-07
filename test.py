from forecasting import *
from plotter import *
from evaluation import *
import pandas as pd


def main():
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    dativendita = best20color(dativendita)
    datetoweek(dativendita)

    # dativendita_nero = filter_by_color(dativendita, 'fantasia')

    train, test = datasplitter(dativendita)
    print("test")

    dativendita_colore = weeksdistrubution(train)

    dativendita_colore_test = weeksdistrubution(test)
    forecast_index = dativendita_colore.index.size - 1
    # plot_dataframe(test_col, plot_name="Test")

    print('Best method: ' + evaluate_simple_forecasts(dativendita_colore, dativendita_colore_test, 'vendite', season=25))

    # testing average
    for i in range(0, 27):
        week_to_forecast = dativendita_colore.index[dativendita_colore.index.size - 1]
        forecast_date, forecast_value = average_forecasting(dativendita_colore['vendite'], week_to_forecast)
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(dativendita_colore)
    plot_dataframe(dativendita_colore,dativendita_colore_test, plot_name="Average", forecasting_indexes=forecast_index)
    print("Average MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    dativendita_colore = weeksdistrubution(train)
    # testing seasonal naive
    for i in range(0, 27):
        week_to_forecast = dativendita_colore.index[dativendita_colore.index.size - 1]
        forecast_date, forecast_value = seasonal_naive_forecasting(dativendita_colore['vendite'], week_to_forecast, 25,1)
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(dativendita_colore)
    plot_dataframe(dativendita_colore, dativendita_colore_test, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)
    print("Seasonal Naive MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    #  Naive
    dativendita_colore = weeksdistrubution(train)
    for i in range(0, 27):
        forecast_date, forecast_value = naive(dativendita_colore, dativendita_colore.index[dativendita_colore.index.size - 1])
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(df, plot_name="Naive")
    plot_dataframe(dativendita_colore,dativendita_colore_test, plot_name="Naive", forecasting_indexes=forecast_index)
    print("Naive MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    # testing drift
    dativendita_colore = weeksdistrubution(train)
    for i in range(0, 27):
        newdf = driftmethod(dativendita_colore)
    plot_dataframe(newdf, dativendita_colore_test, plot_name="Drift", forecasting_indexes=forecast_index)
    print("Drift MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    dativendita_colore = weeksdistrubution(train)
    # testing seasonalexpsmooth
    modelo = seasonalExp_smoothing(dativendita_colore,27)
    plot_dataframe(modelo,dativendita_colore_test, plot_name="HoltWinter", forecasting_indexes=forecast_index)
    print("ETS MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))

    dativendita_colore = weeksdistrubution(train)
    # testing simpleExpSmothing
    forecasted = smpExpSmoth(dativendita_colore, 27)
    plot_dataframe(forecasted, dativendita_colore_test, plot_name='simpleExpSmothing', forecasting_indexes=forecast_index)
    print("Simple ETS MASE: " + str(mase(dativendita_colore['vendite'], dativendita_colore_test['vendite'])))


if __name__ == '__main__':
    main()
