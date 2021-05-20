
from scipy.stats import stats
from forecasting import *
from plotter import plot_dataframe


def main():
    colore = 'giallo'
    df = prepare_data(colore)
    train, test = data_splitter(df, int(len(df.index) * 0.2))
    train = train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]
    test = test[(np.abs(stats.zscore(test)) < 3).all(axis=1)]
    last_week = train.index[train.index.size - 1]
    forecast_index = train.index.size - 1




    if colore== "giallo":
        # Average
        df_avg = average_forecasting(train.copy(), last_week, week_to_forecast=len(test.index))
        # plot_dataframe(df_avg, test, plot_name="Average", forecasting_indexes=forecast_index)

        # Seasonal Naive
        df_sn = seasonal_naive_forecasting(train.copy(), last_week, 26, 1, week_to_forecast=len(test.index))
        # plot_dataframe(df_sn, test, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)

        #  Naive
        df_n = naive(train.copy(), last_week, week_to_forecast=len(test.index))
        # plot_dataframe(df_n, test, plot_name="Naive", forecasting_indexes=forecast_index)

        # Drift
        df_d = driftmethod(train.copy(), last_week, week_to_forecast=len(test.index))
        plot_dataframe(df_d, test, plot_name="Drift", forecasting_indexes=forecast_index)

        # Seasonal Exp Smoothing
        df_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=True)
        plot_dataframe(df_hw, test, plot_name="Holt-Winters", forecasting_indexes=forecast_index)

        # Simple Exp Smoothing
        df_ses = smpExpSmoth(train.copy(), len(test.index))
        plot_dataframe(df_ses, test, plot_name='Simple Exponential Smoothing', forecasting_indexes=forecast_index)
    # TODO


def prepare_data(colore):
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    dativendita = best20color(dativendita)
    datetoweek(dativendita)
    dativendita = filter_by_color(dativendita, colore)
    dativendita_colore = weeksdistrubution(dativendita)
    return dativendita_colore


if __name__ == '__main__':
    main()
