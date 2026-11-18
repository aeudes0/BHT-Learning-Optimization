"""W6 partie B : simulation du prix optimisé sous contrainte de capacité.
Mêmes mondes que presentations/P1/team_analysis.py (un modèle OLS par compétition, prix concurrent rejoués,
demande Poisson), avec en plus le run d'Alois du 2/11. Lancer depuis la racine : venv/Scripts/python.exe W6-Notebooks/w6_simulation.py"""
import os, sys, time, importlib.util, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.join('presentations', 'P1'))
from dpc_files import simon_runs
warnings.filterwarnings('ignore')
ALOIS = os.path.join('BHT-Learning-Optimization_Alois', 'alois_b3c028b', 'results')

runs = [('Simon', 'before 16/10', 'duopoly_competition_details1.csv')]
runs += [('Simon', label, f) for label, _, f in simon_runs()]
for key, label in {'good': 'W2 A', 'bad': 'W2 B', '1021': '21/10', '1024': '24/10', '1102': '2/11'}.items():
    runs.append(('Alois', label, os.path.join(ALOIS, f'duopoly_competition_details_{key}.csv')))
frames = []
for owner, label, f in runs:
    d = pd.read_csv(f); d['owner'], d['run'] = owner, label
    d['competition_id'] = owner[0] + '_' + d['competition_id']; frames.append(d)
df = pd.concat(frames, ignore_index=True)

duo = df[df.competitor_has_capacity]
blocks = range(20, 101, 20)
pooled = {k: sm.OLS(duo[(duo.selling_period > k - 20) & (duo.selling_period <= k)].demand,
                    duo[(duo.selling_period > k - 20) & (duo.selling_period <= k)][['price', 'price_competitor']].assign(intercept=1)).fit().params.to_dict() for k in blocks}
worlds = {}
for comp, dc in duo.groupby('competition_id'):
    worlds[comp] = {}
    for k in blocks:
        d = dc[(dc.selling_period > k - 20) & (dc.selling_period <= k)]
        m = dict(pooled[k]) if len(d) < 100 else sm.OLS(d.demand, d[['price', 'price_competitor']].assign(intercept=1)).fit().params.to_dict()
        m['price'] = min(m['price'], -0.005); worlds[comp][k] = m

def simulate_season(algo, ds, model, rng):
    np.random.seed(int(rng.integers(1e9)))
    ds = ds.sort_values('selling_period')
    pcs, caps = ds.price_competitor.values, ds.competitor_has_capacity.values.astype(bool)
    info, prices, dem, sold = None, [], [], 0
    for t in range(1, 101):
        ph = None if t == 1 else np.array([prices, pcs[:t - 1]])
        price, info = algo.p(1, t, ph, None if t == 1 else np.array(dem), bool(caps[t - 1]), info)
        m = model[next(k for k in blocks if t <= k)]
        pc = pcs[t - 1] if caps[t - 1] else 150.0
        q = min(int(rng.poisson(max(0.0, m['intercept'] + m['price'] * price + m['price_competitor'] * pc))), 80 - sold)
        sold += q; prices.append(price); dem.append(q)
    return np.dot(prices, dem), sold

def evaluate(algo):
    rows = []
    for comp, dc in df.groupby('competition_id'):
        for s in range(51, 101, 2):
            rev, sold = simulate_season(algo, dc[dc.selling_season == s], worlds[comp], np.random.default_rng(s))
            rows.append(dict(competition_id=comp, owner=dc.owner.iloc[0], run=dc.run.iloc[0], season=s, revenue=rev, sold=sold))
    return pd.DataFrame(rows)

def variant(name, **opts):
    spec = importlib.util.spec_from_file_location(name, 'duopoly.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.USE_ARIMA = False                      # même résultat qu'avec ARIMA (±3 €), beaucoup plus rapide
    for k, v in opts.items(): setattr(m, k, v)
    return m

variants = {
    'W4 (actuelle)': variant('w4', USE_CAPACITY_OPT=False),
    'W6 prix optimisé + marges': variant('w6b', USE_CAPACITY_OPT=True, FOLLOW_BAND=True),
    'W6 prix optimisé seul': variant('w6', USE_CAPACITY_OPT=True, FOLLOW_BAND=False),
    'W6 optimisé + marge basse seule': variant('w6d', USE_CAPACITY_OPT=True, FOLLOW_BAND=True, MULTIPLIER_UP=0.0),
    'W4 sans marge haute (témoin)': variant('w4d', USE_CAPACITY_OPT=False, MULTIPLIER_UP=0.0),
}
res = {}
for name, algo in variants.items():
    t0 = time.time(); res[name] = evaluate(algo); print(f'{name} : {time.time() - t0:.0f} s', flush=True)

base = res['W4 (actuelle)']
train = (base.owner == 'Simon') & base.run.isin(['before 16/10', '21/10', '24/10'])
summary = pd.DataFrame({name: {
    'revenu / saison (tout)': r.revenue.mean(),
    'revenu / saison (jamais vu)': r[~train.values].revenue.mean(),
    'sièges vendus': r.sold.mean(),
    'saisons complètes (80 sièges) %': 100 * (r.sold >= 80).mean(),
    'saisons gagnées vs W4 %': 100 * (r.revenue.values > base.revenue.values).mean(),
    'saisons perdues vs W4 %': 100 * (r.revenue.values < base.revenue.values).mean(),
} for name, r in res.items()}).round(1)
print(summary.to_string())
by_run = pd.DataFrame({k: v.groupby(['owner', 'run']).revenue.mean() for k, v in res.items()}).round(0)
print(by_run.to_string())
summary.to_csv('W6-Notebooks/w6_simulation_summary.csv'); by_run.to_csv('W6-Notebooks/w6_simulation_by_run.csv')
pd.concat(res, names=['variant']).to_pickle('W6-Notebooks/w6_simulation.pkl')
