"""W6 partie A : la régression comme problème d'optimisation (Pyomo), avec et sans contraintes."""
import sys, os, time, warnings
import numpy as np, pandas as pd, pyomo.environ as pyo
sys.path.insert(0, os.path.join('presentations', 'P1'))
from dpc_files import simon_runs
warnings.filterwarnings('ignore')

files = ['duopoly_competition_details1.csv'] + [f for l, d, f in simon_runs() if l in ('21/10', '24/10')]
df = pd.concat([pd.read_csv(f) for f in files])
df = df[df.competitor_has_capacity]
HIGHS = pyo.SolverFactory('highs')

def fit(X, y, loss, constrained):
    m = pyo.ConcreteModel()
    n = len(y); m.I = pyo.RangeSet(0, n - 1)
    m.a, m.b, m.c = pyo.Var(), pyo.Var(), pyo.Var()
    if constrained:                       # contraintes métier : prix plus haut = moins de demande, concurrent plus cher = plus
        m.b.setub(-0.005); m.c.setlb(0.0)
    pred = lambda i: m.a + m.b * X[i, 0] + m.c * X[i, 1]
    if loss == 'OLS':
        m.e = pyo.Var(m.I)
        m.r = pyo.Constraint(m.I, rule=lambda m, i: m.e[i] == y[i] - pred(i))
        m.o = pyo.Objective(expr=sum(m.e[i] ** 2 for i in m.I))
    else:                                 # LAD : |e| = e+ + e-
        m.ep = pyo.Var(m.I, domain=pyo.NonNegativeReals); m.em = pyo.Var(m.I, domain=pyo.NonNegativeReals)
        m.r = pyo.Constraint(m.I, rule=lambda m, i: m.ep[i] - m.em[i] == y[i] - pred(i))
        m.o = pyo.Objective(expr=sum(m.ep[i] + m.em[i] for i in m.I))
    HIGHS.solve(m)
    return m.a.value, m.b.value, m.c.value

rows = []
for p_max in range(20, 101, 20):
    blk = df[(df.selling_period > p_max - 20) & (df.selling_period <= p_max)]
    tr, te = blk[blk.selling_season <= 50], blk[blk.selling_season > 50]
    X, y = tr[['price', 'price_competitor']].values, tr.demand.values.astype(float)
    for loss in ('OLS', 'LAD'):
        for cons in (False, True):
            t0 = time.time(); a, b, c = fit(X, y, loss, cons)
            pred = a + b * te.price + c * te.price_competitor
            pc = te.price_competitor.median()
            pstar = np.clip(-(a + c * pc) / (2 * b), 15, 150) if b < 0 else np.nan
            rows.append(dict(days=f'{p_max-19}-{p_max}', model=loss + (' + contraintes' if cons else ''),
                             a=a, b=b, c=c, MAE=np.abs(te.demand - pred).mean(), MSE=((te.demand - pred) ** 2).mean(),
                             p_star=pstar, s=time.time() - t0, n=len(y)))
r = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(r.round(4).to_string())
print('\nzéros dans la demande :', (df.demand == 0).mean().round(3))
r.to_csv('W6-Notebooks/w6_regression_results.csv', index=False)
