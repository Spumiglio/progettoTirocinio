import ast
from dataPreparation import *
from evaluation import *
from forecasting import *
from plotter import *


def main():
    colore = 'rosso'
    cfg = None
    df = prepare_data(colore)
    train, test = data_splitter(df, int(len(df.index) * 0.2))
    forecast_index = train.index.size - 1

    # Naive
    forecast_naive = naive(train.copy(), train.index[train.index.size - 1], week_to_forecast=len(test.index))
    plot_dataframe(forecast_naive, test, plot_name="Naive", forecasting_indexes=forecast_index)

    # Seasonal naive
    forecast_sn = seasonal_naive_forecasting(train.copy(), train.index[train.index.size - 1], 26, 1,
                                             week_to_forecast=len(test.index))
    plot_dataframe(forecast_sn, test, plot_name="Seasonal Naive", forecasting_indexes=forecast_index)

    # Average
    forecast_avg = average_forecasting(train.copy(), train.index[train.index.size - 1],
                                       week_to_forecast=len(test.index))
    plot_dataframe(forecast_avg, test, plot_name="Average", forecasting_indexes=forecast_index)

    # Drift
    forecast_drift = driftmethod(train.copy(), train.index[train.index.size - 1], week_to_forecast=len(test.index))
    plot_dataframe(forecast_drift, test, plot_name="Drift", forecasting_indexes=forecast_index)

    # SES (Simple Exponential Smoothing)
    forecast_ses = smpExpSmoth(train.copy(), len(test.index))
    plot_dataframe(forecast_ses, test, plot_name='Simple Exponential Smoothing', forecasting_indexes=forecast_index)

    if colore == "giallo":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0.1)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0.2)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "avion":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "panna":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "cammello":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "bordeaux":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "senape":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "fango":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "perla":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "cielo":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == "piombo":
        # Seasonal Exp Smoothing
        forecast_hw = seasonalExp_smoothing(train.copy(), len(test.index), decompositon=False, box_cox=0)
        plot_dataframe(forecast_hw, test, plot_name="Holt-Winters: " + colore, forecasting_indexes=forecast_index)
        # Sarima
        scores = grid_search(train['vendite'].copy().values.tolist(), sarima_configs(), n_test=3)
        for cfg, error in scores[:3]:
            pass
        cfg = ast.literal_eval(cfg)
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0)
        plot_dataframe(forecast_sa, test, plot_name='Arima: ' + colore, forecasting_indexes=forecast_index)
    elif colore == 'fantasia':
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

        # Aggregated forecasting
        forecast_driftict = {'forecast_avg': forecast_avg, 'forecast_sn': forecast_sn, 'forecast_naive': forecast_naive,
                             'forecast_drift': forecast_drift,
                             'forecast_hw': forecast_hw, 'forecast_ses': forecast_ses, 'forecast_sa': forecast_sa}
        agg_cfg = best_aggregate_config(forecast_driftict, test)
        cfg_string = str(agg_cfg)
        print("Best Aggregate config: " + cfg_string)

        models = list(agg_cfg)
        df_list = [forecast_driftict[x] for x in models]
        aggregate = aggregate_models(df_list)
        plot_dataframe(aggregate, test, plot_name="Aggregate", forecasting_indexes=forecast_index)
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
        forecast_sa = sarima_forecast(train.copy(), cfg, len(test.index), decomposition=True, box_cox=0.1)
        plot_dataframe(forecast_sa, test, plot_name="SARIMA", forecasting_indexes=train.index.size - 1)

    forecast_driftict = {'forecast_avg': forecast_avg, 'forecast_sn': forecast_sn, 'forecast_naive': forecast_naive,
                         'forecast_drift': forecast_drift,
                         'forecast_hw': forecast_hw, 'forecast_ses': forecast_ses, 'forecast_sa': forecast_sa}
    df_list = [forecast_driftict[x] for x in list(best_aggregate_config(forecast_driftict, test))]
    aggregate = aggregate_models(df_list)
    plot_dataframe(aggregate, test, plot_name="Aggregate: " + colore, forecasting_indexes=forecast_index)


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
