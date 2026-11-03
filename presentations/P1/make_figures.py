"""Figures de la présentation #1 (3 novembre) – à lancer depuis la racine du dépôt :
    venv/Scripts/python.exe presentations/P1/make_figures.py
Les chiffres de simulation viennent de W4-Notebooks/W4_capacity_OLS_ARIMA.ipynb (16 compétitions)
et de la comparaison avec l'algorithme d'Alois (28 compétitions, 31/10 inclus)."""
import glob, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpc_files import simon_runs

OUT = os.path.join('presentations', 'P1', 'figures')
os.makedirs(OUT, exist_ok=True)

BG, INK, MUTED, GRID = '#f7f5f0', '#1f2a37', '#6b7280', '#dedad1'
BLUE, ORANGE, TEAL, GREY = '#2f6fb5', '#e07b39', '#2a9d8f', '#9ca3af'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 15,
    'axes.titlesize': 18, 'axes.titleweight': 'bold', 'legend.frameon': False,
})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('figure', name)

# ---------- données ----------
files = {'before 16/10': 'duopoly_competition_details1.csv'}
for label, _, f in simon_runs():   # dates lues de façon robuste (mois_jour ou jour_mois)
    files[label] = f
frames = [pd.read_csv(f).assign(run=r) for r, f in files.items()]
df = pd.concat(frames, ignore_index=True)
df['key'] = df.competition_id + '_' + df.selling_season.astype(str)
cum = df.pivot_table(values='demand', index='selling_period', columns='key', aggfunc='sum').cumsum()
params = pickle.load(open('duopoly_params.pkl', 'rb'))
band = pd.DataFrame(params['capacity_band']).T

# ---------- 1. résultats DPC ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
runs = ['21/10\n(W3)', '24/10\n(W3)', '31/10\n(W4)']
wins = [33, 67, 83]
ranks = [51, 51, 15]
bars = ax.bar(runs, wins, color=[GREY, GREY, BLUE], width=0.55)
for b, w, r in zip(bars, wins, ranks):
    ax.text(b.get_x() + b.get_width() / 2, w + 2, f'{w}% won', ha='center', fontsize=17, fontweight='bold')
    ax.text(b.get_x() + b.get_width() / 2, 6, f'overall\nrank {r}', ha='center', fontsize=14, color='white')
ax.set_ylim(0, 100)
ax.set_ylabel('competitions won (1st place)')
ax.set_title('DPC results of our algorithm')
save(fig, 'dpc_results.png')

# ---------- 2. courbe cible + marges ----------
train = df[df.run.isin(['before 16/10', '21/10', '24/10'])]   # les 16 compétitions qui ont servi à construire la courbe
rev = train.assign(r=train.price * train.demand).groupby('key').r.sum()
top = rev[rev >= rev.quantile(0.8)].index.intersection(cum.columns)
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(cum.index, cum[top].values / 80, color=GREY, linewidth=0.3, alpha=0.35)
ax.fill_between(band.index, band.lower / 80, band.upper / 80, color=BLUE, alpha=0.25, label='margins (25%–75% quantiles)')
ax.plot(band.index, band.target / 80, color=BLUE, linewidth=3, label='target curve (mean of top 20% seasons)')
ax.set_xlabel('selling period (day)')
ax.set_ylabel('capacity utilization')
ax.set_title('Randomized capacity utilization curve')
ax.legend(loc='upper left')
save(fig, 'capacity_band.png')

# ---------- 3. méthode du cours : courbes aléatoires ----------
mu, std = params['curve_steps']['mu'], params['curve_steps']['std']
rng = np.random.default_rng(0)
days = np.arange(1, 102, 5)
fig, ax = plt.subplots(figsize=(11, 6))
for i in range(40):
    steps = np.clip(rng.normal(mu, std, 20), 0, None)
    c = np.minimum(np.concatenate([[0], np.cumsum(steps)]), 1)
    ax.plot(np.arange(1, 101), np.interp(np.arange(1, 101), days, c), color=ORANGE, linewidth=0.8, alpha=0.5)
ax.fill_between(band.index, band.lower / 80, band.upper / 80, color=BLUE, alpha=0.2, label='our margins')
ax.plot(band.index, band.target / 80, color=BLUE, linewidth=3, label='target curve')
ax.plot([], [], color=ORANGE, label=f'course method: random curves, steps ~ N({mu:.3f}, {std:.3f})')
ax.set_xlabel('selling period (day)')
ax.set_ylabel('capacity utilization')
ax.set_title('Two ways to randomize the target curve')
ax.legend(loc='upper left')
save(fig, 'random_curves.png')

# ---------- 4. courbes cibles vs courbes de ventes ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
for ax, run, title in zip(axes, ['21/10', '24/10', '31/10'], ['21/10 – W3', '24/10 – W3', '31/10 – W4']):
    keys = df[df.run == run].key.unique()
    ax.plot(cum.index, cum[keys].values / 80, color=GREY, linewidth=0.25, alpha=0.25)
    ax.fill_between(band.index, band.lower / 80, band.upper / 80, color=BLUE, alpha=0.25, label='target margins')
    ax.plot(cum.index, cum[keys].mean(axis=1) / 80, color=ORANGE, linewidth=3, label='our booking curve (mean)')
    ax.set_title(title)
    ax.set_xlabel('day')
axes[0].set_ylabel('capacity utilization')
axes[0].legend(loc='upper left', fontsize=13)
fig.suptitle('Target curves vs. booking (sales) curves on the DPC', fontsize=19, fontweight='bold', y=1.03)
save(fig, 'target_vs_booking.png')

# ---------- 5. OLS : une compétition vs modèle commun ----------
old = {20: (-0.0299, 0.0115, 0.9819), 40: (-0.0604, 0.0467, 0.8657), 60: (-0.1449, 0.1353, 0.9659),
       80: (-0.0971, 0.0907, 1.0316), 100: (-0.0794, 0.0588, 1.1548)}
new = {k: (v['price'], v['price_competitor'], v['intercept']) for k, v in params['ols_models'].items()}
pc = np.linspace(5, 100, 50)
def pstar(m, pc):
    b, c, a = m
    return np.clip(-(a + c * pc) / (2 * b), 15, 150)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
for k, ls in [(20, '-'), (60, '--'), (100, ':')]:
    axes[0].plot(pc, pstar(old[k], pc), color=GREY, linestyle=ls, linewidth=2.5, label=f'W3 (1 competition), days {k-19}–{k}')
    axes[0].plot(pc, pstar(new[k], pc), color=BLUE, linestyle=ls, linewidth=2.5, label=f'W4 (16 competitions), days {k-19}–{k}')
axes[0].set_xlabel('competitor price')
axes[0].set_ylabel('optimal price  p* = −(a + c·p_comp) / 2b')
axes[0].set_title('Optimal price from the OLS demand model')
axes[0].legend(fontsize=11, loc='upper left', ncol=2)
axes[0].set_ylim(10, 115)
mae_old = [0.667, 0.833, 1.507, 1.303, 1.141]
mae_new = [0.599, 0.733, 0.722, 0.716, 0.707]
x = np.arange(5)
axes[1].bar(x - 0.2, mae_old, 0.4, color=GREY, label='W3 model (1 competition)')
axes[1].bar(x + 0.2, mae_new, 0.4, color=BLUE, label='W4 pooled model (16 competitions)')
axes[1].set_xticks(x, ['1–20', '21–40', '41–60', '61–80', '81–100'])
axes[1].set_xlabel('days of the season')
axes[1].set_ylabel('mean absolute error (tickets / day)')
axes[1].set_title('Prediction error on test seasons 51–100')
axes[1].set_ylim(0, 2.0)
axes[1].legend(fontsize=12, loc='upper left')
save(fig, 'ols_models.png')

# ---------- 6. ARIMA ----------
ex = df[(df.competition_id == '498Xgm') & (df.selling_season == 60)].sort_values('selling_period')
y = ex.price_competitor.values
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fc = ARIMA(y[:60], order=(1, 1, 1)).fit().get_forecast(15)
ci = fc.conf_int()
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(range(1, 61), y[:60], color=INK, linewidth=2, label='history')
ax.plot(range(61, 101), y[60:], color=GREY, linewidth=2, label='actual')
ax.plot(range(61, 76), fc.predicted_mean, color=ORANGE, linewidth=3, label='ARIMA(1,1,1) forecast')
ax.fill_between(range(61, 76), ci[:, 0], ci[:, 1], color=ORANGE, alpha=0.2, label='95% interval')
ax.set_xlabel('day')
ax.set_ylabel('competitor price')
ax.set_title('Forecasting the competitor price (RustlingNightingale)')
ax.legend(loc='upper left')
save(fig, 'arima.png')

# ---------- 7. simulation : effet de chaque changement ----------
fig, ax = plt.subplots(figsize=(11, 5.8))
names = ['W3 submitted', 'W4 without\nmargins', 'W4 with old\nmonopoly rule', 'W4 + course\nrandom curves', 'W4 final']
vals = [1702, 2067, 1989, 2080, 2096]
bars = ax.bar(names, vals, color=[GREY, BLUE, BLUE, BLUE, ORANGE], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 30, f'{v}', ha='center', fontsize=16, fontweight='bold')
ax.set_ylabel('simulated revenue per season')
ax.set_title('Simulation on 16 competitions: what each change brings')
ax.set_ylim(0, 2400)
save(fig, 'simulation_ablation.png')

# ---------- 9. courbes de ventes des deux membres de l'équipe ----------
ALOIS = os.path.join('BHT-Learning-Optimization_Alois', 'alois_latest')
alois_runs = [('W2 run A', 'good'), ('W2 run B', 'bad'), ('21/10', '1021'), ('24/10', '1024')]
simon_booking = [r for r in files if r != 'before 16/10'][-4:]   # les 4 derniers runs de Simon
alois_curve = pickle.load(open(os.path.join(ALOIS, 'archive', 'week-3-submitted', 'target_sales_curve.pkl'), 'rb'))
ac_days = sorted(int(k) for k in alois_curve)
ac_vals = [alois_curve[k]['sold_seats'] if isinstance(alois_curve[k], dict) else alois_curve[k] for k in sorted(alois_curve, key=int)]
fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), sharey=True)
for ax, run in zip(axes[0], simon_booking):
    keys = df[df.run == run].key.unique()
    ax.plot(cum.index, cum[keys].values / 80, color=GREY, linewidth=0.25, alpha=0.25)
    ax.fill_between(band.index, band.lower / 80, band.upper / 80, color=BLUE, alpha=0.25, label='Simon’s target margins')
    ax.plot(cum.index, cum[keys].mean(axis=1) / 80, color=ORANGE, linewidth=3, label='booking curve (mean)')
    ax.set_title(f'Simon – {run}')
for ax, (label, key) in zip(axes[1], alois_runs):
    a = pd.read_csv(os.path.join(ALOIS, 'results', f'duopoly_competition_details_{key}.csv'))
    a['key'] = a.competition_id + '_' + a.selling_season.astype(str)
    ac = a.pivot_table(values='demand', index='selling_period', columns='key', aggfunc='sum').cumsum()
    ax.plot(ac.index, ac.values / 80, color=GREY, linewidth=0.25, alpha=0.25)
    ax.plot(ac_days, np.array(ac_vals) / 80, color=TEAL, linewidth=3, linestyle='--', label='Alois’ target curve')
    ax.plot(ac.index, ac.mean(axis=1) / 80, color=ORANGE, linewidth=3, label='booking curve (mean)')
    ax.set_title(f'Alois – {label}')
    ax.set_xlabel('day')
axes[0][0].legend(loc='upper left', fontsize=12)
axes[1][0].legend(loc='upper left', fontsize=12)
axes[0][0].set_ylabel('capacity utilization')
axes[1][0].set_ylabel('capacity utilization')
fig.tight_layout()
save(fig, 'team_booking.png')

# ---------- 10. comparaison équitable des deux algorithmes (presentations/P1/team_analysis.py) ----------
sim = pd.read_csv(os.path.join('presentations', 'P1', 'team_simulation.csv'), index_col=0)
cols = list(sim.columns[1:])   # Simon (n), Alois (n), jamais vues (n)
num = [c.split('(')[-1].rstrip(')') for c in cols]
labels = [f'Simon’s {num[0]}\ncompetitions', f'Alois’ {num[1]}\ncompetitions', f'{num[2]} competitions unseen\nby Simon W4’s model']
algos = [('Simon W3', '#c3c8cf'), ('Alois W3', '#9fd3cb'), ('Simon W4', BLUE), ('Alois W4', TEAL)]
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cols))
for i, (name, color) in enumerate(algos):
    vals = sim.loc[name, cols].values
    bars = ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=color, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f'{v:.0f}', ha='center', fontsize=11)
ax.set_xticks(x, labels)
ax.set_ylabel('simulated revenue per season')
ax.set_title(f'Same simulator, {sim.columns[0].split()[1]} competitions from both of us')
ax.legend(ncol=4, loc='upper left', fontsize=12)
ax.set_ylim(0, 3400)
save(fig, 'team_fair_comparison.png')

# ---------- 8. comparaison des deux algorithmes de l'équipe (ancienne version, 28 compétitions de Simon) ----------
fig, ax = plt.subplots(figsize=(11, 5.8))
groups = ['before 16/10', '21/10', '24/10', '31/10', 'all 28']
simon_w3 = [1972, 2579, 596, 468, 1163]
alois_w4 = [1860, 2949, 576, 482, 1228]
simon_w4 = [2262, 3317, 612, 470, 1366]
x = np.arange(len(groups))
ax.bar(x - 0.27, simon_w3, 0.27, color=GREY, label='Simon W3 (offline OLS, 1 competition)')
ax.bar(x, alois_w4, 0.27, color=TEAL, label='Alois W4 (online ARX within the season)')
ax.bar(x + 0.27, simon_w4, 0.27, color=BLUE, label='Simon W4 (pooled OLS + margins)')
ax.set_xticks(x, groups)
ax.set_xlabel('competitions (by DPC run)')
ax.set_ylabel('simulated revenue per season')
ax.set_title('Same simulator, two team approaches')
ax.legend(fontsize=12, loc='upper right')
save(fig, 'team_comparison.png')
