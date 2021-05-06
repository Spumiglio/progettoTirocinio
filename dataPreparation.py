from math import log
from math import exp
from forecasting import *
from datetime import *
from datetime import timedelta
import dateutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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
    for i in range(0, 11):
        venditetemp.popitem()
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

def datasplitter(dativendita, testsize=0.2):
    train, test = train_test_split(dativendita, test_size=testsize, random_state=42, shuffle=False)
    return train, test


def box_cox_transformation(df, lambda_num, reverse=False):
    for index in df.index:
        if not reverse:
            if lambda_num == 0:
                df.loc[index] = log(df.loc[index])
            else:
                df.loc[index] = np.sign(df.loc[index]) * ((abs(df.loc[index])**lambda_num - 1) / lambda_num)
        else:
            if lambda_num == 0:
                df.loc[index] = exp(df.loc[index])
            else:
                df.loc[index] = np.sign(lambda_num * df.loc[index] + 1) * (abs(lambda_num * df.loc[index] + 1)**(1/lambda_num))

    return df




