"""Analyse équilibrée des deux algorithmes de l'équipe (Simon et Alois) – à lancer depuis la racine du dépôt :
    venv/Scripts/python.exe presentations/P1/team_analysis.py
Données : CSV DPC de Simon (premier CSV + résultat_DPC/) et d'Alois (branche alois, dossier results/,
extrait dans BHT-Learning-Optimization_Alois/alois_latest/)."""
import glob, os, pickle, importlib.util, warnings, time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dpc_files import simon_runs
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)

ALOIS = os.path.join('BHT-Learning-Optimization_Alois', 'alois_latest')
OUT = os.path.join('presentations', 'P1')

# ---------- données des deux équipes ----------
runs = [('Simon', 'before 16/10', 'duopoly_competition_details1.csv')]
for label, _, f in simon_runs():   # dates lues de façon robuste (mois_jour ou jour_mois)
    runs.append(('Simon', label, f))
alois_names = {'good': 'W2 run A (good)', 'bad': 'W2 run B (bad)', '1021': '21/10', '1024': '24/10'}
for key, label in alois_names.items():
    runs.append(('Alois', label, os.path.join(ALOIS, 'results', f'duopoly_competition_details_{key}.csv')))
frames = []
for owner, label, f in runs:
    d = pd.read_csv(f)
    d['owner'], d['run'] = owner, label
    d['competition_id'] = owner[0] + '_' + d['competition_id']   # évite les collisions d'identifiants
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df['revenue'] = df.price * df.demand
df['key'] = df.competition_id + '_' + df.selling_season.astype(str)

# ---------- statistiques DPC par run ----------
season = df.groupby(['owner', 'run', 'key']).agg(rev=('revenue', 'sum'), sold=('demand', 'sum'),
                                                 conc_out=('competitor_has_capacity', lambda x: (~x).sum()))
stats = season.groupby(['owner', 'run']).agg(competitions=('rev', lambda x: x.index.get_level_values('key').str.rsplit('_', n=1).str[0].nunique()),
                                             revenue_per_season=('rev', 'mean'), seats_sold=('sold', 'mean'),
                                             our_stockout=('sold', lambda x: (x >= 80).mean() * 100),
                                             competitor_stockout=('conc_out', lambda x: (x > 0).mean() * 100))
mono = df[~df.competitor_has_capacity].groupby(['owner', 'run']).agg(monopoly_tickets=('demand', 'sum'),
                                                                     monopoly_price=('price', 'mean'))
stats = stats.join(mono).join(df.groupby(['owner', 'run']).price.mean().rename('mean_price'))
print(stats.round(1).to_string())
stats.round(1).to_csv(os.path.join(OUT, 'team_dpc_stats.csv'))

# ---------- simulateur : un monde par compétition ----------
duo = df[df.competitor_has_capacity]
pooled = {}
for p_max in range(20, 101, 20):
    d = duo[(duo.selling_period > p_max - 20) & (duo.selling_period <= p_max)]
    X = d[['price', 'price_competitor']].assign(intercept=1)
    pooled[p_max] = sm.OLS(d.demand, X).fit().params.to_dict()
worlds = {}
for comp, dc in duo.groupby('competition_id'):
    worlds[comp] = {}
    for p_max in range(20, 101, 20):
        d = dc[(dc.selling_period > p_max - 20) & (dc.selling_period <= p_max)]
        if len(d) < 100:
            m = dict(pooled[p_max])
        else:
            m = sm.OLS(d.demand, d[['price', 'price_competitor']].assign(intercept=1)).fit().params.to_dict()
        m['price'] = min(m['price'], -0.005)
        worlds[comp][p_max] = m

def simulate_season(algo, ds, model, rng):
    np.random.seed(int(rng.integers(1e9)))
    ds = ds.sort_values('selling_period')
    pcs, caps = ds.price_competitor.values, ds.competitor_has_capacity.values.astype(bool)
    info, prices, dem, sold = None, [], [], 0
    for t in range(1, 101):
        ph = None if t == 1 else np.array([prices, pcs[:t - 1]])
        dh = None if t == 1 else np.array(dem)
        price, info = algo.p(1, t, ph, dh, bool(caps[t - 1]), info)
        m = model[next(k for k in (20, 40, 60, 80, 100) if t <= k)]
        pc = pcs[t - 1] if caps[t - 1] else 150.0
        q = min(int(rng.poisson(max(0.0, m['intercept'] + m['price'] * price + m['price_competitor'] * pc))), 80 - sold)
        sold += q
        prices.append(price)
        dem.append(q)
    return np.dot(prices, dem), sold

def evaluate(algo, seasons=range(51, 101, 2)):
    rows = []
    for comp, dc in df.groupby('competition_id'):
        for s in seasons:
            rev, sold = simulate_season(algo, dc[dc.selling_season == s], worlds[comp], np.random.default_rng(s))
            rows.append({'competition_id': comp, 'owner': dc.owner.iloc[0], 'run': dc.run.iloc[0], 'season': s,
                         'revenue': rev, 'sold': sold})
    return pd.DataFrame(rows)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def int_keys(path):
    curve = pickle.load(open(path, 'rb'))
    return {int(k): v for k, v in curve.items()}

simon_w3 = load('simon_w3', 'archive/week-3-submitted/duopoly_2026-10-20.py')
simon_w4 = load('simon_w4', 'duopoly.py')
simon_w4.USE_ARIMA = False   # même résultat qu'avec ARIMA (±3 €), beaucoup plus rapide
alois_w3 = load('alois_w3', os.path.join(ALOIS, 'archive/week-3-submitted/duopoly.py'))
curve_w3 = int_keys(os.path.join(ALOIS, 'archive/week-3-submitted/target_sales_curve.pkl'))
alois_w3.load_target_curve = lambda: curve_w3
alois_w4 = load('alois_w4', os.path.join(ALOIS, 'archive/week-4-submitted/duopoly.py'))
curve_w4 = int_keys(os.path.join(ALOIS, 'archive/week-4-submitted/target_sales_curve.pkl'))
alois_w4.load_target_curve = lambda: curve_w4   # clés converties en entiers (le .pkl a des clés texte)

res = {}
for name, algo in [('Simon W3', simon_w3), ('Alois W3', alois_w3), ('Simon W4', simon_w4), ('Alois W4', alois_w4)]:
    t0 = time.time()
    res[name] = evaluate(algo)
    print(name, 'simulé en %.0f s' % (time.time() - t0))

# mondes jamais vus par le modèle OLS de Simon W4 (entraîné sur ses 16 premières compétitions)
trained_on = ['before 16/10', '21/10', '24/10']
base = res['Simon W4']
unseen = ~((base.owner == 'Simon') & base.run.isin(trained_on))
n = lambda mask: base[mask.values].competition_id.nunique()
n_all, n_s, n_a, n_u = n(base.owner.notna()), n(base.owner == 'Simon'), n(base.owner == 'Alois'), n(unseen)
summary = pd.DataFrame({
    f'all {n_all} competitions': {k: v.revenue.mean() for k, v in res.items()},
    f'Simon competitions ({n_s})': {k: v[v.owner == 'Simon'].revenue.mean() for k, v in res.items()},
    f'Alois competitions ({n_a})': {k: v[v.owner == 'Alois'].revenue.mean() for k, v in res.items()},
    f'unseen by Simon W4 model ({n_u})': {k: v[unseen.values].revenue.mean() for k, v in res.items()},
}).round(0)
print(summary.to_string())
summary.to_csv(os.path.join(OUT, 'team_simulation.csv'))
by_run = pd.DataFrame({k: v.groupby(['owner', 'run']).revenue.mean() for k, v in res.items()}).round(0)
print(by_run.to_string())
by_run.to_csv(os.path.join(OUT, 'team_simulation_by_run.csv'))
for a, b in [('Simon W4', 'Alois W4'), ('Simon W3', 'Alois W3')]:
    w = (res[a].revenue.values > res[b].revenue.values)
    t = (res[a].revenue.values == res[b].revenue.values)
    print(f'{a} > {b} : {100 * w.mean():.0f}% des saisons (égalité {100 * t.mean():.0f}%) | mondes jamais vus : {100 * w[unseen.values].mean():.0f}%')
pickle.dump(res, open(os.path.join(OUT, 'team_simulation.pkl'), 'wb'))
