from forecasting import *
from plotter import *
import pandas as pd

def main():
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    best20color(dativendita)
    datetoweek(dativendita)
    dativendita_colore = weeksdistrubution(dativendita)
    dflist = dataframelist(dativendita)
    df_col = weeksdistrubution(dflist[0])

    # simpleExpSmothing
    for i in range(0, 12):
        smpExpsmoth(df_col)
    plot_dataframe(df_col, plot_name='simpleExpSmothing')


    # testing average
    for i in range(0, 12):
        week_to_forecast = dativendita_colore.index[dativendita_colore.index.size - 1]
        forecast_date, forecast_value = average_forecasting(dativendita_colore['vendite'], week_to_forecast)
        dativendita_colore.loc[forecast_date] = forecast_value
    plot_dataframe(dativendita_colore)

    # testing seasonal naive
    for i in range(0, 100):
        week_to_forecast = dativendita_colore.index[dativendita_colore.index.size - 1]
        forecast_date, forecast_value = seasonal_naive_forecasting(dativendita_colore['vendite'], week_to_forecast, 25,1)
        dativendita_colore.loc[forecast_date] = forecast_value
    plot_dataframe(dativendita_colore)

    #  Naive
    df = weeksdistrubution(dativendita)
    for i in range(0, 12):
        forecast_date, forecast_value = naive(df['vendite'], df.index[df.index.size - 1])
        df.loc[forecast_date] = forecast_value
    plot_dataframe(df, plot_name="Naive")

    for df in dflist:
        df_col = weeksdistrubution(df)
        # plotting
        plot_dataframe(df_col, plot_name=df.iloc[0, 16])

    # testing drift
    for i in range(0, 12):
        newdf = driftmethod(df_col)
    plot_dataframe(newdf, plot_name="Drift")



if __name__ == '__main__':
    main()
