import datetime
import dateutil

import pandas as pd
import statsmodels
import matplotlib.pyplot as plt
import numpy as np
from datetime import *
from datetime import timedelta


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
    val = []
    venditetemp = {}
    count=1
    for row in dativendita.itertuples():
        count = 1
        for i in range(0, 12):
            data = datetime.fromisoformat(row[16])
            if data.isocalendar()[1]+i > 52:
                weekStr = str(data.isocalendar()[0]+1) + "-W" + str(count)

                count +=1
                if weekStr in venditetemp.keys():
                    venditetemp[weekStr] += row[i + 3]
                else:
                    venditetemp[weekStr] = row[i + 3]
            else:
                weekStr = str(data.isocalendar()[0]) + "-W" + str(data.isocalendar()[1]+i)

                if weekStr in venditetemp.keys():
                    venditetemp[weekStr] += row[i+3]
                else:
                    venditetemp[weekStr] = row[i + 3]


    for key in list(venditetemp.keys()):
        val.append(venditetemp[key])

    timeseries = pd.DataFrame(index=list(venditetemp.keys()))
    timeseries.insert(0, "vendite", val, allow_duplicates=True)
    return timeseries

def weeksdistrubution2(dativendita):
    val = []
    venditetemp = {}

    for row in dativendita.itertuples():
        for i in range(0, 12):
            weekStr = addWeek(row[15],i)
            if weekStr in venditetemp.keys():
                    venditetemp[weekStr] += row[i+3]
            else:
                    venditetemp[weekStr] = row[i + 3]

    for key in list(venditetemp.keys()):
        val.append(venditetemp[key])

    timeseries = pd.DataFrame(index=list(venditetemp.keys()))
    timeseries.insert(0, "vendite", val, allow_duplicates=True)
    return timeseries


def addWeek(date_string, weeks):
    date_iso = dateutil.parser.isoparse(date_string)
    new_date = date_iso + timedelta(weeks=weeks)
    new_date_iso = str(datetime.fromisoformat(new_date.isoformat()).isocalendar()[0]) + "-W" + \
           str(datetime.fromisoformat(new_date.isoformat()).isocalendar()[1])
    return new_date_iso

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

    print("2020-W51: " + addWeek("2020-W51", 1))
    print("2020-W52: " + addWeek("2020-W52", 1))
    print("2020-W53: " + addWeek("2020-W53", 1))
    print("2021-W01: " + addWeek("2021-W01", 1))
    print("2019-W36: " + addWeek("2019-W36", 1))
    print("2016-W1: " + addWeek("2016-W1", 1))
    print("2016-W9: " + addWeek("2016-W9", 1))


if __name__ == '__main__':
    main()
