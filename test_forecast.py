import ast

from scipy.stats import stats

from dataPreparation import data_splitter, sommavendite, best20color, datetoweek, filter_by_color, weeksdistrubution
from evaluation import grid_search, sarima_configs
from forecasting import *
from plotter import plot_dataframe


def main():
    colore = 'fantasia'

    df = prepare_data(colore)
    train, test = data_splitter(df, int(len(df.index) * 0.2))
    last_week = train.index[train.index.size - 1]
    forecast_index = train.index.size - 1

    if colore == "giallo":
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

    if colore == 'fantasia':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=True, box_cox=0.1)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0.1)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'nero':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'rosa':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=True, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'bianco':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0.1,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0,
                                      rmv_outliers=False)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'jeans':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'beige':
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=True, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'grigio':
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=False, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'verde militare':
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0.1,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        # TODO continua a dare valori negativi: scegliere il lambda giusto
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'rosso':
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0,
                                      rmv_outliers=False)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'verde':
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0.1,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        # TODO non va bene!
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0.1,
                                      rmv_outliers=True)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)


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
