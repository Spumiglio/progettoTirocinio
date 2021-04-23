import pandas as pd
import statsmodels
import matplotlib.pyplot as plt
import numpy as np

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

def main():
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    filtrato = best20color(dativendita)


    '''plt.figure()
    dativendita.plot.area(x="giorno_uscita",y="somma_vendite",alpha=0.5)
    plt.show()'''


if __name__ == '__main__':
    main()
