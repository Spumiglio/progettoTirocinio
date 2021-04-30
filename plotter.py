# type_of_plot puo' essere H = istogramma, B = barre, L = lineare
import matplotlib.pyplot as plt

def plot_dataframe(df, type_of_plot="L", plot_name="Vendite totali", forecasting_indexes=0):
    if type_of_plot == "L":
        x = df.index
        y = df['vendite']
        plt.plot(x, y)
        plt.xlabel("Settimane")
        plt.ylabel("Totale vendite")
        plt.title(plot_name)
        plt.show()


