import numpy as np
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from tabulate import tabulate
from dataPreparation import sommavendite, best20color, datetoweek, filter_by_color, weeksdistrubution, \
    fill_missing_data, data_splitter
from plotter import *
from evaluation import *
import ast
from scipy import stats


def main():
    colore = 'fantasia'
    cfg = None
    train = prepare_data(colore)
    forecast_index = train.index.size - 1
    week_to_forecast = 27

    # Naive
    forecast_naive = naive(train.copy(), train.index[train.index.size - 1], week_to_forecast)
    plot_dataframe(forecast_naive, plot_name="Naive", forecasting_indexes=forecast_index)

    # Seasonal naive
    forecast_sn = seasonal_naive_forecasting(train.copy(), train.index[train.index.size - 1], 26, 1,
                                             week_to_forecast)
    plot_dataframe(forecast_sn, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)

    # Average
    forecast_avg = average_forecasting(train.copy(), train.index[train.index.size - 1],
                                       week_to_forecast)
    plot_dataframe(forecast_avg, plot_name="Average", forecasting_indexes=forecast_index)

    # Drift
    forecast_drift = driftmethod(train.copy(), train.index[train.index.size - 1], week_to_forecast)
    plot_dataframe(forecast_drift, plot_name="Drift", forecasting_indexes=forecast_index)

    # SES (Simple Exponential Smoothing)
    forecast_ses = smpExpSmoth(train.copy(), week_to_forecast)
    plot_dataframe(forecast_ses, plot_name='Simple Exponential Smoothing', forecasting_indexes=forecast_index)

    if colore == "giallo":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0.1)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0.2)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "avion":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0.1)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0.2)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "panna":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "cammello":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "bordeaux":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0.1)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg,week_to_forecast, decomposition=True, box_cox=0.1)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "senape":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(),week_to_forecast, decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "fango":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg,week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "perla":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "cielo":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0.1)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "piombo":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(),week_to_forecast, decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg,week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == 'fantasia':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=True, box_cox=0.1)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0.1)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'nero':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'rosa':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=True, box_cox=0.1)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'bianco':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0.2)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'jeans':
        # Holt-Winters
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0.1,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        # SARIMA
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'beige':
        forecast_hw = seasonalExp_smoothing(train.copy(),week_to_forecast, decompositon=True, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'grigio':
        forecast_hw = seasonalExp_smoothing(train.copy(),week_to_forecast, decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg,week_to_forecast, decomposition=False, box_cox=0)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'verde militare':
        forecast_hw = seasonalExp_smoothing(train.copy(),week_to_forecast, decompositon=False, box_cox=0.1,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg,week_to_forecast, decomposition=True, box_cox=0,
                                      rmv_outliers=False)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'rosso':
        forecast_hw = seasonalExp_smoothing(train.copy(),week_to_forecast, decompositon=False, box_cox=0,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0,
                                      rmv_outliers=False)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)
    elif colore == 'verde':
        forecast_hw = seasonalExp_smoothing(train.copy(), week_to_forecast, decompositon=False, box_cox=0.1,
                                            rmv_outliers=False)
        plot_dataframe(forecast_hw, plot_name="Holt-Winters", forecasting_indexes=train.index.size - 1)

        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, week_to_forecast, decomposition=True, box_cox=0.1)
        plot_dataframe(forecast_sa, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)

    forecast_driftict = {'forecast_avg': forecast_avg, 'forecast_sn': forecast_sn,
                         'forecast_naive': forecast_naive, 'forecast_drift': forecast_drift,
                         'forecast_hw': forecast_hw, 'forecast_ses': forecast_ses, 'forecast_sa': forecast_sa}
    aggregate = aggregate_models([forecast_hw, forecast_drift])
    plot_dataframe(aggregate, plot_name="Aggregate: " + colore, forecasting_indexes=forecast_index)

    weight = {'forecast_hw': 0.4, 'forecast_sa': 0.25, 'forecast_drift': 0.16, 'forecast_ses': 0.1, 'forecast_naive': 0.05, 'forecast_avg': 0.03, 'forecast_sn': 0.01}
    weighted = aggregate_weighted(weight, forecast_driftict, week_to_forecast)
    plot_dataframe(weighted, plot_name="Aggregate Weighted: " + colore, forecasting_indexes=forecast_index)


def prepare_data(colore):
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    dativendita = best20color(dativendita)
    datetoweek(dativendita)
    dativendita = filter_by_color(dativendita, colore)
    dativendita_colore = weeksdistrubution(dativendita)
    dativendita_colore = fill_missing_data(dativendita_colore, start=dativendita_colore.index[0],
                                           end=dativendita_colore.index[len(dativendita_colore.index) - 1],
                                           fill_mode='V', fill_value=0)
    return dativendita_colore



if __name__ == '__main__':
    main()
