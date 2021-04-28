import datetime

import dateutil
from dateutil import parser
import pandas as pd
import statsmodels
import matplotlib.pyplot as plt
import numpy as np
from datetime import *
from datetime import timedelta


def filter_by_color(df, color):
    df = df[df['colore'] == color]
    return df

def dataframelist(df):
    dflist=[]
    colorlist=best20colorlist(df)
    for color in colorlist:
        dflist.append(filter_by_color(df, color))
    return dflist

def best20color(dativendita):
    rag = dativendita.groupby(by="colore").sum().sort_values(by=["somma_vendite"], ascending=False).head(20)
    index = rag.index.to_series(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    filtered = dativendita[dativendita.colore.isin(index)]
    return filtered


def best20colorlist(dativendita):
    rag = dativendita.groupby(by="colore").sum().sort_values(by=["somma_vendite"], ascending=False).head(20)
    list = rag.index.values.tolist()
    return list


def sommavendite(dativendita):
    somma_vendite = dativendita["0"] + dativendita["1"] + dativendita["2"] + dativendita["3"] + \
                    dativendita["4"] + dativendita["5"] + dativendita["6"] + dativendita["7"] + \
                    dativendita["8"] + dativendita["9"] + dativendita["10"] + dativendita["11"]
    dativendita.insert(1, "somma_vendite", somma_vendite, allow_duplicates=True)
    return dativendita


def datetoweek(dativendita):
    dataweek = dativendita["giorno_uscita"]
    weeks = []
    for datastr in dataweek:
        data = datetime.fromisoformat(datastr)
        week = str(data.isocalendar()[0]) + "-W" + str(data.isocalendar()[1])
        weeks.append(week)

    weekserie = pd.Series(weeks, dativendita.index)
    dativendita.insert(14, "settimana", weekserie, allow_duplicates=True)
    return dativendita


def weeksdistrubution(dativendita):
    val = []
    venditetemp = {}
    for row in dativendita.itertuples():
        for i in range(0, 12):
            weekStr = add_week(row[15], i)
            if weekStr in venditetemp.keys():
                venditetemp[weekStr] += row[i + 3]
            else:
                venditetemp[weekStr] = row[i + 3]
    for key in list(venditetemp.keys()):
        val.append(venditetemp[key])
    timeseries = pd.DataFrame(index=list(venditetemp.keys()))
    timeseries.insert(0, "vendite", val, allow_duplicates=True)
    return timeseries


def add_week(date_string, weeks):
    date_iso = dateutil.parser.isoparse(date_string)
    new_date = date_iso + timedelta(weeks=weeks)
    new_date_iso = str(datetime.fromisoformat(new_date.isoformat()).isocalendar()[0]) + "-W" + \
                   str(datetime.fromisoformat(new_date.isoformat()).isocalendar()[1])
    return new_date_iso


# type_of_plot puo' essere H = istogramma, B = barre, L = lineare
def plot_dataframe(df, type_of_plot="L", plot_name="Vendite totali"):
    if type_of_plot == "L":
        x = df.index
        y = df['vendite']
        plt.plot(x, y)
        plt.xlabel("Settimane")
        plt.ylabel("Totale vendite")
        plt.title(plot_name)
        plt.show()

def main():
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    best20color(dativendita)
    datetoweek(dativendita)
    dativendita = filter_by_color(dativendita, "nero")
    dativendita_colore = weeksdistrubution(dativendita)
    dflist = dataframelist(dativendita)

    # plotting
    plot_dataframe(dativendita_colore)


if __name__ == '__main__':
    main()
