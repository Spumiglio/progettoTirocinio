import pandas as pd

from dataPreparation import sommavendite, best20color, datetoweek, filter_by_color, weeksdistrubution


def main():
    colore = 'fantasia'

    df = prepare_data(colore)

    # TODO


def prepare_data(colore):
    dativendita = pd.read_csv("students_dataset_attr.csv").sort_values(by=["giorno_uscita"])
    dativendita = sommavendite(dativendita)
    dativendita = best20color(dativendita)
    datetoweek(dativendita)

    dativendita = filter_by_color(dativendita, 'nero')
    dativendita_colore = weeksdistrubution(dativendita)
    return dativendita_colore


if __name__ == '__main__':
    main()
