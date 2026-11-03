"""Liste des CSV DPC de Simon avec une étiquette de date fiable.

Les noms de fichiers n'ont pas toujours le même format : 'details10_21_2026' (mois_jour)
mais 'details2_11_2026' (jour_mois). On essaie les deux lectures et on garde la seule date
plausible : pendant le cours (octobre 2026 - février 2027) et pas dans le futur."""
import datetime as dt
import glob
import os
import re

COURSE_START, COURSE_END = dt.date(2026, 10, 1), dt.date(2027, 2, 28)


def run_date(path, today=None):
    today = today or dt.date.today()
    a, b, year = map(int, re.search(r'details(\d+)_(\d+)_(\d+)\.csv$', os.path.basename(path)).groups())
    dates = set()
    for month, day in ((a, b), (b, a)):
        try:
            d = dt.date(year, month, day)
        except ValueError:
            continue
        if COURSE_START <= d <= min(COURSE_END, today):
            dates.add(d)
    if len(dates) != 1:
        raise ValueError(f'date ambiguë ou impossible pour {path}: {sorted(dates)}')
    return dates.pop()


def simon_runs(folder='résultat_DPC'):
    """[(étiquette 'jj/mm', date, chemin)] triés par date."""
    out = [(run_date(f), f) for f in glob.glob(os.path.join(folder, '*.csv'))]
    return [(f'{d.day}/{d.month:02d}', d, f) for d, f in sorted(out)]


if __name__ == '__main__':
    for label, d, f in simon_runs():
        print(label, d, f)
