"""W7 : simulation de l'approche d'apprentissage (Bertsimas & Perakis) dans duopoly.py.
Nouveautés par rapport au simulateur des semaines 4-6 :
- les saisons d'une même compétition sont jouées À LA SUITE, avec le même information_dump
  (comme sur DPC), pour que l'algorithme puisse apprendre la demande de la compétition ;
- mode 'reactive' : le concurrent réagit à notre prix comme dans les données
  (prix_conc(t) - prix_conc(t-1) = alpha + k*(notre_prix(t-1) - prix_conc(t-1)), k estimé par compétition) :
  son prix simulé = prix rejoué + D(t), avec D(t) = (1-k)*D(t-1) + k*(notre prix simulé - notre prix réel)(t-1) ;
- mode 'replay' : prix concurrent rejoués tels quels (comme avant), pour vérifier la robustesse.
Lancer depuis la racine : venv/Scripts/python.exe W7-Notebooks/w7_simulation.py"""
import os, sys, time, zlib, importlib.util, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'presentations', 'P1'))
from dpc_files import simon_runs

SEASONS = range(1, 31)          # 30 saisons jouées à la suite par compétition
BLOCKS = (20, 40, 60, 80, 100)
ALOIS = os.path.join(ROOT, 'BHT-Learning-Optimization_Alois', 'alois_b3c028b', 'results')
OUT = os.path.join(ROOT, 'W7-Notebooks')


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def load_data():
    runs = [('Simon', 'before 16/10', os.path.join(ROOT, 'duopoly_competition_details1.csv'))]
    runs += [('Simon', label, f if os.path.isabs(f) else os.path.join(ROOT, f)) for label, _, f in simon_runs(os.path.join(ROOT, 'résultat_DPC'))]
    for key, label in {'good': 'W2 A', 'bad': 'W2 B', '1021': '21/10', '1024': '24/10', '1102': '2/11'}.items():
        runs.append(('Alois', label, os.path.join(ALOIS, f'duopoly_competition_details_{key}.csv')))
    comps = []
    for owner, label, f in runs:
        df = pd.read_csv(f).sort_values(['competition_id', 'selling_season', 'selling_period'])
        for cid, dc in df.groupby('competition_id'):
            duo = dc[dc.competitor_has_capacity]
            X = np.column_stack([np.ones(len(duo)), duo.price, duo.price_competitor])
            world = {}
            for k in BLOCKS:
                m = (duo.selling_period > k - 20).values & (duo.selling_period <= k).values
                beta = ols(X[m], duo.demand.values[m]) if m.sum() >= 100 else ols(X, duo.demand.values)
                world[k] = (beta[0], min(beta[1], -0.005), beta[2])
            # réaction du concurrent, estimée sur les données réelles
            g = dc.groupby('selling_season')
            gap = (dc.price - dc.price_competitor).groupby(dc.selling_season).shift(1)
            change = g.price_competitor.diff()
            ok = gap.notna() & change.notna() & dc.competitor_has_capacity & g.competitor_has_capacity.shift(1).fillna(False).astype(bool)
            k_react = 0.0
            if ok.sum() > 30 and gap[ok].var() > 0:
                k_react = float(np.clip(np.cov(gap[ok], change[ok])[0, 1] / gap[ok].var(), 0, 1))
            seasons = {s: (ds.price.values, ds.price_competitor.values, ds.competitor_has_capacity.values.astype(bool))
                       for s, ds in dc.groupby('selling_season') if s in SEASONS and len(ds) == 100}
            comps.append(dict(owner=owner, run=label, cid=owner[0] + '_' + cid, world=world, k=k_react, seasons=seasons))
    return comps


def simulate_competition(algo, comp, reactive):
    rng = np.random.default_rng(zlib.crc32(comp["cid"].encode()))
    np.random.seed(int(rng.integers(1e9)))
    info, rows = None, []
    for s in SEASONS:
        if s not in comp['seasons']:
            continue
        p_log, pc_log, caps = comp['seasons'][s]
        prices, pcs, dem, sold, D = [], [], [], 0, 0.0
        for t in range(1, 101):
            if reactive and t > 1:
                D = (1 - comp['k']) * D + comp['k'] * (prices[-1] - p_log[t - 2])
            pc = float(np.clip(pc_log[t - 1] + (D if reactive else 0.0), 0.0, 150.0))
            ph = None if t == 1 else np.array([prices, pcs])
            price, info = algo.p(s, t, ph, None if t == 1 else np.array(dem, dtype=float), bool(caps[t - 1]), info)
            a, b, c = comp['world'][next(k for k in BLOCKS if t <= k)]
            pc_eff = pc if caps[t - 1] else 150.0
            q = min(int(rng.poisson(max(0.0, a + b * price + c * pc_eff))), 80 - sold)
            sold += q; prices.append(price); pcs.append(pc); dem.append(q)
        rows.append(dict(cid=comp['cid'], owner=comp['owner'], run=comp['run'], k=comp['k'], season=s,
                         revenue=float(np.dot(prices, dem)), sold=sold))
    return rows


VARIANTS = {
    'W4 (sur DPC)': dict(USE_CAPACITY_OPT=False),
    'W6': dict(),
    'W7 demande apprise (niveau)': dict(USE_ONLINE_DEMAND=True, ONLINE_MODE='level'),
    'W7 demande apprise (a, b, c)': dict(USE_ONLINE_DEMAND=True, ONLINE_MODE='full'),
    'W7 réaction seule': dict(USE_REACTION=True),
    'W7 niveau + réaction': dict(USE_ONLINE_DEMAND=True, ONLINE_MODE='level', USE_REACTION=True),
    'W7 niveau + réaction, sans marges': dict(USE_ONLINE_DEMAND=True, ONLINE_MODE='level', USE_REACTION=True, FOLLOW_BAND=False),
}


def load_algo(name, opts):
    spec = importlib.util.spec_from_file_location('algo_' + str(abs(hash(name))), os.path.join(ROOT, 'duopoly.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.USE_ARIMA = False      # même résultat qu'avec ARIMA (±3 €), beaucoup plus rapide
    for k, v in opts.items():
        setattr(m, k, v)
    return m


_COMPS = None
def job(args):
    global _COMPS
    name, reactive = args
    if _COMPS is None:
        _COMPS = load_data()
    algo = load_algo(name, VARIANTS[name])
    t0 = time.time(); rows = []
    for comp in _COMPS:
        for r in simulate_competition(algo, comp, reactive):
            r['variant'], r['mode'] = name, 'reactive' if reactive else 'replay'
            rows.append(r)
    return rows, name, reactive, time.time() - t0


if __name__ == '__main__':
    jobs = [(n, r) for r in (True, False) for n in VARIANTS]
    rows = []
    with Pool(min(len(jobs), max(1, os.cpu_count() - 2))) as pool:
        for r, name, reactive, sec in pool.imap_unordered(job, jobs):
            rows += r
            print(f'{name} ({"réactif" if reactive else "rejoué"}) : {sec:.0f} s', flush=True)
    res = pd.DataFrame(rows)
    res.to_pickle(os.path.join(OUT, 'w7_simulation.pkl'))
    pd.set_option('display.width', 250)
    order = list(VARIANTS)
    for mode in ('reactive', 'replay'):
        r = res[res['mode'] == mode]
        base = r[r.variant == 'W4 (sur DPC)'].set_index(['cid', 'season']).revenue
        tab = {}
        for v in order:
            x = r[r.variant == v]
            xs = x.set_index(['cid', 'season']).revenue
            tab[v] = {
                'revenu / saison': x.revenue.mean(),
                'saisons 11-30': x[x.season > 10].revenue.mean(),
                'concurrents suiveurs (k>0,3)': x[x.k > 0.3].revenue.mean(),
                'concurrents indépendants (k<0,05)': x[x.k < 0.05].revenue.mean(),
                'sièges vendus': x.sold.mean(),
                '% saisons > W4': 100 * (xs > base.reindex(xs.index)).mean(),
                '% saisons < W4': 100 * (xs < base.reindex(xs.index)).mean(),
            }
        tab = pd.DataFrame(tab).round(1)
        print(f'\n=== concurrent {"réactif" if mode == "reactive" else "rejoué (ne réagit pas)"} ===')
        print(tab.to_string())
        tab.to_csv(os.path.join(OUT, f'w7_summary_{mode}.csv'))
        by_run = r.pivot_table(index=['owner', 'run'], columns='variant', values='revenue', aggfunc='mean')[order].round(0)
        by_run.to_csv(os.path.join(OUT, f'w7_by_run_{mode}.csv'))
    print('\n', res[res['mode'] == 'reactive'].pivot_table(index=['owner', 'run'], columns='variant', values='revenue', aggfunc='mean')[order].round(0).to_string())
