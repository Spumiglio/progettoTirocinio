import matplotlib.pyplot as plt


# type_of_plot puo' essere H = istogramma, B = barre, L = lineare
def plot_dataframe(df, test = 0, type_of_plot="L", plot_name="Vendite totali", forecasting_indexes=0):
    if type_of_plot == "L":
        df_temp = df.copy()
        x = df_temp.index
        historic = df_temp['vendite'].loc[df_temp['vendite'].index[0]:df_temp['vendite'].index[forecasting_indexes]]
        forecasting = df_temp['vendite'].loc[df_temp['vendite'].index[forecasting_indexes]:df_temp['vendite'].index[-1]]
        df_temp.insert(1, "forecasting", forecasting)
        df_temp['vendite'] = historic
        y = df_temp['vendite']
        f = df_temp['forecasting']
        plt.plot(x, y, 'r',  f, 'b')
        plt.plot(test)
        plt.xlabel("Settimane")
        plt.ylabel("Totale vendite")
        plt.legend(["Train","Forecast","Test"])
        plt.title(plot_name)
        plt.show()

