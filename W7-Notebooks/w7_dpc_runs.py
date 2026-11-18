"""Analyse des runs DPC de Simon (stats par run, réaction du concurrent). Lancer depuis la racine."""
import sys, os, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.join('presentations', 'P1'))
from dpc_files import simon_runs
rows = []; react = []
for label, d, f in simon_runs():
    df = pd.read_csv(f); df['rev'] = df.price * df.demand
    s = df.groupby(['competition_id', 'selling_season']).agg(rev=('rev', 'sum'), sold=('demand', 'sum'))
    mono = df[~df.competitor_has_capacity]
    comp = s.groupby('competition_id').sold.mean()
    rows.append(dict(run=label, comps=df.competition_id.nunique(), rev=s.rev.mean(), sold=s.sold.mean(),
                     stockout=(s.sold >= 80).mean() * 100, mean_price=df.price.mean(), floor15=(df.price <= 15.01).mean() * 100,
                     mono_tickets=mono.demand.sum(), mono_price=mono.price.mean(),
                     sold_min=comp.min(), sold_max=comp.max()))
    # le concurrent réagit-il à notre prix d'hier ?
    df = df.sort_values(['competition_id', 'selling_season', 'selling_period'])
    g = df.groupby(['competition_id', 'selling_season'])
    df['p_lag'] = g.price.shift(1); df['pc_lag'] = g.price_competitor.shift(1)
    x = df[df.competitor_has_capacity].dropna(subset=['p_lag', 'pc_lag'])
    for cid, xc in x.groupby('competition_id'):
        m = sm.OLS(xc.price_competitor - xc.pc_lag, sm.add_constant(xc.p_lag - xc.pc_lag)).fit()
        react.append(dict(run=label, comp=cid, coef=m.params.iloc[1], t=m.tvalues.iloc[1],
                          corr_same_day=np.corrcoef(xc.price, xc.price_competitor)[0, 1]))
pd.set_option('display.width', 220)
print(pd.DataFrame(rows).round(1).to_string())
r = pd.DataFrame(react)
print('\nréaction du concurrent à notre écart de prix d hier (pc_t - pc_t-1 = k * (p_t-1 - pc_t-1)) :')
print('k médian %.3f | compétitions avec |t|>3 : %d / %d | k>0.05 et t>3 : %d' % (r.coef.median(), (r.t.abs() > 3).sum(), len(r), ((r.coef > 0.05) & (r.t > 3)).sum()))
print(r.coef.describe().round(3).to_string())
