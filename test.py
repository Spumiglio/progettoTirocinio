import pandas as pd
import statsmodels
import matplotlib.pyplot as plt
import numpy as np
from datetime import *


def best20color(dativendita):
    rag = dativendita.groupby(by="colore").sum().sort_values(by=["somma_vendite"], ascending=False).head(20)
    index = rag.index.to_series(index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    filtered = dativendita[dativendita.colore.isin(index)]
    return filtered


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
    timeseries = pd.DataFrame(index=dativendita["settimana"])
    venditetemp = {}
    for row in dativendita.itertuples():
        for i in range(0, 12):
            print(datetime.fromisoformat(row["settimana"]))
            '''if row["settimana"] in venditetemp.keys():
                venditetemp[datetime.fromisoformat(row["settimana"])+i] += row[str(i)]
            else:
                venditetemp[datetime.fromisoformat(row["settimana"])+i] = row[str(i)]'''

    # timeseries.insert(1,"vendite",questobuisognatrovaerlo,allow_duplicates=True)

    return timeseries


def main():
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    best20color(dativendita)
    datetoweek(dativendita)
    weeksdistrubution(dativendita)
    # print(dativendita.head(5))
    '''plt.figure()
    dativendita.plot.area(x="giorno_uscita",y="somma_vendite",alpha=0.5)
    plt.show()'''


if __name__ == '__main__':
    main()
