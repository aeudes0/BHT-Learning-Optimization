"""W7 étapes 1-2 sur les vraies données DPC : quelle estimation prédit le mieux la demande du jour suivant ?
- 'commun'           : modèle commun du .pkl (semaine 4), sans apprentissage ;
- 'cours (LAD lisse)': méthode du cours : LAD avec des beta qui changent chaque jour (|beta_t - beta_t+1| <= delta),
                       estimée sur les jours précédents de la saison en cours (programme linéaire, scipy linprog / HiGHS),
                       puis moyenne des beta des N = 10 derniers jours ;
- 'niveau appris'    : duopoly.py W7, ONLINE_MODE = 'level' (saisons précédentes de la même compétition) ;
- 'a, b, c appris'   : duopoly.py W7, ONLINE_MODE = 'full'.
Lancer depuis la racine : venv/Scripts/python.exe W7-Notebooks/w7_demand_learning.py"""
import os, sys, importlib.util, warnings
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join('presentations', 'P1'))
from dpc_files import simon_runs

DELTA, N_AVG = 0.05, 10


def lad_smooth(d, p1, p2, delta=DELTA):
    """Programme linéaire du cours (étape 1). Variables : beta0_t, beta1_t, beta2_t (t = 1..T) puis Delta_t.
    min sum Delta_t  s.c.  |d_t - (b0_t + b1_t p1_t + b2_t p2_t)| <= Delta_t,  |b_t^i - b_t+1^i| <= delta,
    b0 >= 0, b1 <= 0, b2 >= 0 (signes du code du prof)."""
    T = len(d); nb = 3 * T; nv = nb + T
    A = lil_matrix((2 * T + 6 * (T - 1), nv)); ub = np.zeros(2 * T + 6 * (T - 1)); r = 0
    for t in range(T):
        for sgn in (1, -1):        # sgn*(d - pred) <= Delta  <=>  -sgn*pred - Delta <= -sgn*d
            A[r, 3 * t] = -sgn; A[r, 3 * t + 1] = -sgn * p1[t]; A[r, 3 * t + 2] = -sgn * p2[t]; A[r, nb + t] = -1
            ub[r] = -sgn * d[t]; r += 1
    for t in range(T - 1):
        for i in range(3):
            for sgn in (1, -1):
                A[r, 3 * t + i] = sgn; A[r, 3 * (t + 1) + i] = -sgn; ub[r] = delta; r += 1
    bounds = [(0, None), (None, 0), (0, None)] * T + [(0, None)] * T
    res = linprog(np.r_[np.zeros(nb), np.ones(T)], A_ub=A.tocsr(), b_ub=ub, bounds=bounds, method='highs')
    if not res.success:
        return None
    beta = res.x[:nb].reshape(T, 3)
    return beta[-N_AVG:].mean(axis=0)


def load_algo(mode):
    spec = importlib.util.spec_from_file_location('d_' + mode, 'duopoly.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.USE_ONLINE_DEMAND, m.ONLINE_MODE = True, mode
    return m


if __name__ == '__main__':
    level, full = load_algo('level'), load_algo('full')
    rng = np.random.default_rng(0)
    rows = []
    runs = [x for x in simon_runs() if x[1] >= pd.Timestamp('2026-11-04').date()]   # runs jamais vus par le modèle commun
    for label, _, f in runs:
        df = pd.read_csv(f).sort_values(['competition_id', 'selling_season', 'selling_period'])
        for cid in rng.choice(df.competition_id.unique(), 3, replace=False):
            dc = df[df.competition_id == cid]
            ll, lf = level.learn_init(), full.learn_init()
            for s in range(1, 13):                       # 12 saisons : on apprend sur 1..11, on teste 6, 9, 12
                ds = dc[dc.selling_season == s]
                d, p1, p2, cap = ds.demand.values, ds.price.values, ds.price_competitor.values, ds.competitor_has_capacity.values
                if s in (6, 9, 12):
                    for t in range(20, 100, 4):          # prédire la demande du jour t (indice t-1)
                        m = cap[:t - 1]
                        if not cap[t - 1] or m.sum() < 15:
                            continue
                        x = np.array([1.0, p1[t - 1], p2[t - 1]])
                        beta = lad_smooth(d[:t - 1][m], p1[:t - 1][m], p2[:t - 1][m])
                        preds = {'commun': x @ np.array(level.get_ols_params(t)),
                                 'niveau appris': x @ np.array(level.get_ols_params(t, ll)),
                                 'a, b, c appris': x @ np.array(full.get_ols_params(t, lf)),
                                 'cours (LAD lisse)': np.nan if beta is None else x @ beta}
                        for k, v in preds.items():
                            rows.append(dict(run=label, cid=cid, season=s, t=t, model=k, err=d[t - 1] - v, demand=d[t - 1]))
                for t in range(1, 101):                  # apprentissage de la saison (comme dans duopoly.py)
                    if cap[t - 1] and ds.demand.values[:t].sum() < 80:
                        for mod, st in ((level, ll), (full, lf)):
                            mod.learn_update(st, t, p1[t - 1], p2[t - 1], d[t - 1])
        print(label, 'ok', flush=True)
    r = pd.DataFrame(rows)
    tab = r.groupby('model').agg(MSE=('err', lambda e: np.mean(e ** 2)), MAE=('err', lambda e: np.mean(np.abs(e))),
                                 biais=('err', 'mean'), n=('err', 'size')).round(3)
    print(tab.sort_values('MSE').to_string())
    tab.to_csv(os.path.join('W7-Notebooks', 'w7_demand_learning.csv'))
