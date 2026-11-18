"""Analyse des runs DPC de Simon (stats par run, réaction du concurrent). Lancer depuis la racine."""
import sys, os, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.join('presentations', 'P1'))
from dpc_files import simon_runs
out = []
for label, d, f in simon_runs():
    df = pd.read_csv(f).sort_values(['competition_id', 'selling_season', 'selling_period'])
    g = df.groupby(['competition_id', 'selling_season'])
    df['dpc'] = g.price_competitor.diff(); df['dp_lag'] = g.price.diff().groupby([df.competition_id, df.selling_season]).shift(1)
    df['dp_lag2'] = df.groupby(['competition_id', 'selling_season']).dp_lag.shift(1)
    df['dp_same'] = g.price.diff(); df['dpc_lag'] = df.groupby(['competition_id', 'selling_season']).dpc.shift(1)
    df['dp_lead'] = df.groupby(['competition_id', 'selling_season']).dp_same.shift(-1)
    x = df[df.competitor_has_capacity].dropna(subset=['dpc', 'dp_lag', 'dpc_lag', 'dp_same', 'dp_lead', 'dp_lag2'])
    for cid, xc in x.groupby('competition_id'):
        if xc.dp_lag.std() == 0 or xc.dpc.std() == 0: continue
        m = sm.OLS(xc.dpc, sm.add_constant(xc[['dp_lag', 'dp_lag2', 'dpc_lag', 'dp_same', 'dp_lead']])).fit()
        out.append(dict(run=label, comp=cid, k1=m.params.dp_lag, t1=m.tvalues.dp_lag, k2=m.params.dp_lag2,
                        same=m.params.dp_same, t_same=m.tvalues.dp_same, lead=m.params.dp_lead, t_lead=m.tvalues.dp_lead))
r = pd.DataFrame(out)
print(len(r), 'compétitions')
for c in ['k1', 'k2', 'same', 'lead']:
    print(f'{c:5s} médiane {r[c].median():.3f}  moyenne {r[c].mean():.3f}  q75 {r[c].quantile(.75):.3f}')
print('notre changement d hier significatif (t>3) :', (r.t1 > 3).sum(), '| placebo changement de demain (t>3) :', (r.t_lead > 3).sum(), '| même jour (t>3):', (r.t_same > 3).sum())
print(r.sort_values('k1', ascending=False).head(12).round(2).to_string())
r['type'] = pd.cut(r.k1, [-9, 0.05, 0.3, 0.95, 9], labels=['indépendant', 'suiveur faible', 'suiveur fort', 'copieur'])
print(r.groupby('run').type.value_counts().unstack().to_string())
print(r.type.value_counts())
