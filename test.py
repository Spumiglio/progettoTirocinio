from forecasting import *
from plotter import *
import pandas as pd

def main():
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    best20color(dativendita)
    datetoweek(dativendita)
    train, test = datasplitter(dativendita)
    dativendita_colore = weeksdistrubution(train)
    dflist = dataframelist(train)
    df_col = weeksdistrubution(dflist[0])

    dativendita_colore_test = weeksdistrubution(test)
    testlist = dataframelist(test)
    test_col = weeksdistrubution(testlist[0])


    forecast_index = dativendita_colore.index.size - 1
    plot_dataframe(test_col, plot_name="Test")

    for df in dflist:
        df_col = weeksdistrubution(df)
        # plotting
        # plot_dataframe(df_col, plot_name=df.iloc[0, 16])

    # testing average
    for i in range(0, 27):
        week_to_forecast = dativendita_colore.index[dativendita_colore.index.size - 1]
        forecast_date, forecast_value = average_forecasting(dativendita_colore['vendite'], week_to_forecast)
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(dativendita_colore)
    plot_dataframe(dativendita_colore, plot_name="Average", forecasting_indexes=forecast_index)

    dativendita_colore = weeksdistrubution(train)
    # testing seasonal naive
    for i in range(0, 27):
        week_to_forecast = dativendita_colore.index[dativendita_colore.index.size - 1]
        forecast_date, forecast_value = seasonal_naive_forecasting(dativendita_colore['vendite'], week_to_forecast, 25,1)
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(dativendita_colore)
    plot_dataframe(dativendita_colore, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)

    #  Naive
    dativendita_colore = weeksdistrubution(train)
    for i in range(0, 27):
        forecast_date, forecast_value = naive(dativendita_colore, dativendita_colore.index[dativendita_colore.index.size - 1])
        dativendita_colore.loc[forecast_date] = forecast_value
    # plot_dataframe(df, plot_name="Naive")
    plot_dataframe(dativendita_colore, plot_name="Naive", forecasting_indexes=forecast_index)

    # testing drift
    dativendita_colore = weeksdistrubution(train)
    for i in range(0, 27):
        newdf = driftmethod(dativendita_colore)
    plot_dataframe(newdf, plot_name="Drift", forecasting_indexes=forecast_index)

    dativendita_colore = weeksdistrubution(train)
    # testing seasonalexpsmooth
    modelo = seasonalExp_smoothing(dativendita_colore,27)
    plot_dataframe(modelo, plot_name="HoltWinter", forecasting_indexes=forecast_index)

    dativendita_colore = weeksdistrubution(train)
    # testing simpleExpSmothing
    smpExpSmoth(dativendita_colore, 27)
    plot_dataframe(dativendita_colore, plot_name='simpleExpSmothing', forecasting_indexes=forecast_index)


if __name__ == '__main__':
    main()
