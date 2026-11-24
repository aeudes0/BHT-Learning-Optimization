# %% [1] IMPORTS, REAL duopoly.p LOADER & MULTI-FILE CSV LOADER
import glob
import importlib
import itertools
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import duopoly
importlib.reload(duopoly)  # Always pick up your latest edits to duopoly.py

DATA_DIR = os.path.join(BASE_DIR, "results")
CAPACITY_PER_SEASON = 80
MAX_RUNS_ON_LEARNING_PLOT = 4


def load_all_runs(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "duopoly_competition_details_*.csv")
    files = sorted(glob.glob(pattern))

    print(f"Looking in: {os.path.abspath(data_dir)}")
    print(f"Total CSV files found and imported: {len(files)}")

    if not files:
        raise FileNotFoundError(f"No matching CSV files found in {os.path.abspath(data_dir)}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        run_id = os.path.splitext(os.path.basename(f))[0].replace("duopoly_competition_details_", "")
        df["run_id"] = run_id
        df["revenue"] = df["price"] * df["demand"]
        frames.append(df)
        print(f"  - Imported {os.path.basename(f)} | {len(df):,} rows | {df['competition_id'].nunique()} competitions")

    return pd.concat(frames, ignore_index=True)


raw_df = load_all_runs(DATA_DIR)

# %% [2] SEASON-LEVEL AGGREGATION & SUMMARY TABLES
def build_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["active_comp_price"] = np.where(df["competitor_has_capacity"], df["price_competitor"], np.nan)
    df["price_diff"] = df["price"] - df["active_comp_price"]

    seasons = df.groupby(["run_id", "competition_id", "competitor_id", "selling_season"], as_index=False).agg(
        total_revenue=("revenue", "sum"),
        total_demand=("demand", "sum"),
        avg_price=("price", "mean"),
        avg_comp_price=("active_comp_price", "mean"),
        avg_price_diff=("price_diff", "mean"),
        comp_stockout_periods=("competitor_has_capacity", lambda s: (~s).sum()),
        max_calc_ms=("calculation_duration", lambda s: s.max() * 1000),
        error_count=("errors", lambda s: s.notna().sum()),
    )
    seasons["load_factor"] = seasons["total_demand"] / CAPACITY_PER_SEASON
    seasons["realized_yield"] = np.where(
        seasons["total_demand"] > 0,
        seasons["total_revenue"] / seasons["total_demand"],
        0.0,
    )
    return seasons


season_df = build_season_summary(raw_df)

run_leaderboard = season_df.groupby("run_id").agg(
    competitions=("competition_id", "nunique"),
    mean_rev=("total_revenue", "mean"),
    median_rev=("total_revenue", "median"),
    load_factor_pct=("load_factor", lambda s: s.mean() * 100),
    avg_yield=("realized_yield", "mean"),
    our_avg_price=("avg_price", "mean"),
    max_calc_ms=("max_calc_ms", "max"),
    errors=("error_count", "sum"),
).sort_index()

print("\n=== SUMMARY ACROSS ALL IMPORTED RUNS ===")
print(run_leaderboard.round(2).to_string())

latest_run = sorted(season_df["run_id"].unique())[-1]
opp_breakdown = (
    season_df[season_df["run_id"] == latest_run]
    .groupby("competitor_id")
    .agg(
        seasons=("selling_season", "count"),
        mean_rev=("total_revenue", "mean"),
        load_factor_pct=("load_factor", lambda s: s.mean() * 100),
        our_price=("avg_price", "mean"),
        comp_price=("avg_comp_price", "mean"),
        comp_stockout_periods=("comp_stockout_periods", "mean"),
    )
    .sort_values("mean_rev", ascending=False)
)

print(f"\n=== OPPONENT BREAKDOWN (LATEST RUN: {latest_run}) ===")
print(opp_breakdown.round(2).to_string())

# %% [3] CLEAN, UNCLUTTERED DIAGNOSTIC CHARTS (2x2 GRID)
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax1 = axes[0, 0]
run_means = run_leaderboard["mean_rev"]
bars = ax1.bar(run_means.index.astype(str), run_means.values, color="#2b5c8f", width=0.55)
ax1.set_title("Mean Season Revenue by Run", fontsize=12, fontweight="bold")
ax1.set_xlabel("Run ID")
ax1.set_ylabel("Mean Revenue ($)")
ax1.tick_params(axis="x", rotation=30)
for bar in bars:
    h = bar.get_height()
    ax1.annotate(
        f"${h:,.0f}",
        xy=(bar.get_x() + bar.get_width() / 2, h),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

ax2 = axes[0, 1]
recent_runs = sorted(season_df["run_id"].unique())[-MAX_RUNS_ON_LEARNING_PLOT:]
learning = (
    season_df[season_df["run_id"].isin(recent_runs)]
    .groupby(["run_id", "selling_season"])["total_revenue"]
    .mean()
    .unstack(0)
    .rolling(5, min_periods=1)
    .mean()
)
for col in learning.columns:
    lw = 2.5 if col == latest_run else 1.4
    alpha = 1.0 if col == latest_run else 0.65
    ax2.plot(learning.index, learning[col], label=f"Run {col}", linewidth=lw, alpha=alpha)

ax2.set_title(f"5-Season Rolling Revenue (Latest {len(recent_runs)} Runs)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Selling Season (1–100)")
ax2.set_ylabel("Mean Revenue ($)")
ax2.legend(frameon=True)

ax3 = axes[1, 0]
latest_raw = raw_df[raw_df["run_id"] == latest_run].copy()
latest_raw["cum_demand"] = latest_raw.groupby(["competition_id", "selling_season"])["demand"].cumsum()
burn = latest_raw.groupby("selling_period")["cum_demand"].agg(
    mean="mean",
    p25=lambda s: s.quantile(0.25),
    p75=lambda s: s.quantile(0.75),
)
ax3.plot(burn.index, burn["mean"], label="Mean Cumulative Sales", color="#1b4965", linewidth=2.5)
ax3.fill_between(burn.index, burn["p25"], burn["p75"], color="#62b6cb", alpha=0.25, label="25th–75th Percentile")
ax3.plot([1, 100], [0.8, 80], "--", color="#d9534f", linewidth=1.5, label="Ideal Linear Pace (0.8/period)")
ax3.set_title(f"Intraseason Inventory Burn (Run {latest_run})", fontsize=12, fontweight="bold")
ax3.set_xlabel("Selling Period (1–100)")
ax3.set_ylabel("Cumulative Units Sold")
ax3.set_ylim(0, 85)
ax3.legend(frameon=True)

ax4 = axes[1, 1]
price_traj = latest_raw.groupby("selling_period").agg(
    our_price=("price", "mean"),
    comp_price=("price_competitor", lambda s: s[s < 900].mean()),
)
ax4.plot(price_traj.index, price_traj["our_price"], label="Our Avg Price", color="#2a9d8f", linewidth=2.5)
ax4.plot(price_traj.index, price_traj["comp_price"], label="Competitor Avg Price", color="#e76f51", linestyle="--", linewidth=2.0)
ax4.set_title(f"Intraseason Price Curve (Run {latest_run})", fontsize=12, fontweight="bold")
ax4.set_xlabel("Selling Period (1–100)")
ax4.set_ylabel("Price ($)")
ax4.legend(frameon=True)

plt.tight_layout()
plt.show()

# %% [4] EMPIRICAL DEMAND MODEL & DIRECT BACKTESTER USING YOUR REAL duopoly.p()
class EmpiricalDemandEnvironment:
    """Precomputes a dense 4D NumPy array for O(1) scalar demand rate lookups."""

    def __init__(self, df: pd.DataFrame):
        valid = df.copy()
        valid["price_bin"] = pd.cut(valid["price"], bins=[-1e9, 20, 30, 40, 50, 65, 85, 1e9], labels=False)
        valid["diff_bin"] = pd.cut(
            np.where(valid["competitor_has_capacity"], valid["price"] - valid["price_competitor"], -999.0),
            bins=[-1e9, -100, -15, -5, 5, 15, 1e9],
            labels=False,
        )
        valid["period_bin"] = pd.cut(valid["selling_period"], bins=[-1e9, 25, 50, 75, 1e9], labels=False)

        active_fallback = float(valid[valid["competitor_has_capacity"]]["demand"].mean())
        self.table = np.full((2, 4, 7, 6), active_fallback, dtype=np.float64)

        price_midpoints = [15.0, 25.0, 35.0, 45.0, 57.5, 75.0, 110.0]
        for p_idx, p_mid in enumerate(price_midpoints):
            self.table[0, :, p_idx, :] = max(0.05, 1.5 * np.exp(-0.035 * p_mid))

        grouped = (
            valid.groupby(["competitor_has_capacity", "period_bin", "price_bin", "diff_bin"], observed=False)["demand"]
            .mean()
            .dropna()
        )
        for (cap, t_b, p_b, d_b), val in grouped.items():
            self.table[int(cap), int(t_b), int(p_b), int(d_b)] = float(val)

    def expected_demand(self, price: float, comp_price: float, comp_has_cap: bool, period: int) -> float:
        if price < 20.0: p_bin = 0
        elif price < 30.0: p_bin = 1
        elif price < 40.0: p_bin = 2
        elif price < 50.0: p_bin = 3
        elif price < 65.0: p_bin = 4
        elif price < 85.0: p_bin = 5
        else: p_bin = 6

        if not comp_has_cap:
            d_bin = 0
        else:
            diff = price - comp_price
            if diff < -100.0: d_bin = 0
            elif diff < -15.0: d_bin = 1
            elif diff < -5.0: d_bin = 2
            elif diff < 5.0: d_bin = 3
            elif diff < 15.0: d_bin = 4
            else: d_bin = 5

        if period <= 25: t_bin = 0
        elif period <= 50: t_bin = 1
        elif period <= 75: t_bin = 2
        else: t_bin = 3

        return float(self.table[1 if comp_has_cap else 0, t_bin, p_bin, d_bin])


def backtest_duopoly_p(
    eval_df: pd.DataFrame,
    opponent_mode: str = "historical",
    n_competitions: int = 6,
    seed: int = 42,
    copycat_initial_price: float = None,
    train_df: pd.DataFrame = None,
    env: EmpiricalDemandEnvironment = None,
) -> pd.DataFrame:
    """Runs your ACTUAL `duopoly.p(...)` function across competitions and seasons,
    preserving `information_dump` across seasons and invoking the period 101 callback.
    """
    if opponent_mode not in ("historical", "copycat"):
        raise ValueError("opponent_mode must be 'historical' or 'copycat'")

    if env is None:
        env = EmpiricalDemandEnvironment(train_df if train_df is not None else eval_df)

    rng = np.random.default_rng(seed)
    poisson = rng.poisson
    comp_ids = eval_df["competition_id"].unique()[:n_competitions]
    sub = eval_df[eval_df["competition_id"].isin(comp_ids)].sort_values(
        ["competition_id", "selling_season", "selling_period"]
    )

    results = []
    for comp_id, comp_df in sub.groupby("competition_id", sort=False):
        info_dump = None  # Persists across all seasons of this competition_id
        for season, s_df in comp_df.groupby("selling_season", sort=True):
            opp_id = s_df["competitor_id"].iloc[0]
            comp_price_arr = s_df["price_competitor"].to_numpy(dtype=np.float64)
            comp_cap_arr = s_df["competitor_has_capacity"].to_numpy(dtype=bool)
            rev_arr = s_df["revenue"].to_numpy(dtype=np.float64)

            our_inv = CAPACITY_PER_SEASON
            comp_inv = CAPACITY_PER_SEASON
            our_rev = 0.0
            comp_rev = 0.0
            our_prices, comp_prices, our_demands = [], [], []
            max_step_ms = 0.0

            for step_idx in range(len(s_df)):
                period = step_idx + 1
                logged_cp = comp_price_arr[step_idx]

                if opponent_mode == "historical":
                    comp_has_cap = bool(comp_cap_arr[step_idx])
                    comp_price = logged_cp
                else:
                    comp_has_cap = comp_inv > 0
                    if not comp_has_cap:
                        comp_price = 999.0
                    elif period == 1:
                        if copycat_initial_price is not None:
                            comp_price = float(copycat_initial_price)
                        elif np.isfinite(logged_cp) and logged_cp < 500.0:
                            comp_price = logged_cp
                        else:
                            comp_price = 45.0
                    else:
                        comp_price = our_prices[-1]

                p_hist = np.array([our_prices, comp_prices], dtype=np.float64) if period > 1 else None
                d_hist = np.array(our_demands, dtype=np.float64) if period > 1 else None

                t_call0 = time.perf_counter()
                raw_price, info_dump = duopoly.p(
                    current_selling_season=int(season),
                    selling_period_in_current_season=period,
                    prices_historical_in_current_season=p_hist,
                    demand_historical_in_current_season=d_hist,
                    competitor_has_capacity_current_period_in_current_season=comp_has_cap,
                    information_dump=info_dump,
                )
                step_ms = (time.perf_counter() - t_call0) * 1000.0
                if step_ms > max_step_ms:
                    max_step_ms = step_ms

                if our_inv <= 0:
                    our_price = 150.0
                    our_demand = 0
                else:
                    our_price = float(np.clip(raw_price, 1.0, 150.0))
                    lam_our = env.expected_demand(our_price, comp_price, comp_has_cap, period)
                    d = int(poisson(lam_our))
                    our_demand = our_inv if d > our_inv else d

                if opponent_mode == "copycat" and comp_has_cap:
                    lam_comp = env.expected_demand(comp_price, our_price, our_inv > 0, period)
                    cd = int(poisson(lam_comp))
                    comp_demand = comp_inv if cd > comp_inv else cd
                    comp_inv -= comp_demand
                    comp_rev += comp_price * comp_demand

                our_inv -= our_demand
                our_rev += our_price * our_demand
                our_prices.append(our_price)
                comp_prices.append(comp_price)
                our_demands.append(our_demand)

            # End-of-season period 101 callback (matches DPC server behavior)
            _, info_dump = duopoly.p(
                current_selling_season=int(season),
                selling_period_in_current_season=101,
                prices_historical_in_current_season=np.array([our_prices, comp_prices], dtype=np.float64),
                demand_historical_in_current_season=np.array(our_demands, dtype=np.float64),
                competitor_has_capacity_current_period_in_current_season=comp_inv > 0,
                information_dump=info_dump,
            )

            results.append({
                "competition_id": comp_id,
                "competitor_id": opp_id,
                "selling_season": int(season),
                "opponent_mode": opponent_mode,
                "sim_revenue": our_rev,
                "sim_sold": CAPACITY_PER_SEASON - our_inv,
                "max_calc_ms": max_step_ms,
                "comp_sim_revenue": comp_rev if opponent_mode == "copycat" else np.nan,
                "actual_logged_revenue": float(rev_arr.sum()),
            })

    return pd.DataFrame(results)


# %% [5] TUNE PARAMETERS DIRECTLY THROUGH YOUR REAL duopoly.p() FUNCTION
def tune_duopoly_parameters(
    eval_df: pd.DataFrame,
    param_grid: dict,
    opponent_mode: str = "both",
    copycat_weight: float = 0.5,
    n_competitions: int = 6,
    seed: int = 42,
    train_df: pd.DataFrame = None,
    use_pyomo_in_grid: bool = False,
) -> pd.DataFrame:
    """Mutates the constants directly on the imported `duopoly` module and runs
    your actual `duopoly.p(...)` function on every parameter combination.
    """
    if opponent_mode not in ("historical", "copycat", "both"):
        raise ValueError("opponent_mode must be 'historical', 'copycat', or 'both'")

    shared_env = EmpiricalDemandEnvironment(train_df if train_df is not None else eval_df)

    # Save original constants from duopoly.py so they are cleanly restored afterward
    saved_attrs = {key.upper(): getattr(duopoly, key.upper()) for key in param_grid.keys()}
    saved_pyomo_flag = getattr(duopoly, "ENABLE_PYOMO", True)

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"Tuning {total} combinations directly through duopoly.p() | mode='{opponent_mode}' | pyomo_in_grid={use_pyomo_in_grid}")

    leaderboard = []
    try:
        if hasattr(duopoly, "ENABLE_PYOMO"):
            duopoly.ENABLE_PYOMO = use_pyomo_in_grid

        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            for k, v in params.items():
                setattr(duopoly, k.upper(), float(v))

            hist_rev, hist_sold = np.nan, np.nan
            copy_rev, copy_sold = np.nan, np.nan

            if opponent_mode in ("historical", "both"):
                res_h = backtest_duopoly_p(
                    eval_df, opponent_mode="historical",
                    n_competitions=n_competitions, seed=seed, env=shared_env,
                )
                hist_rev = float(res_h["sim_revenue"].mean())
                hist_sold = float(res_h["sim_sold"].mean())

            if opponent_mode in ("copycat", "both"):
                res_c = backtest_duopoly_p(
                    eval_df, opponent_mode="copycat",
                    n_competitions=n_competitions, seed=seed, env=shared_env,
                )
                copy_rev = float(res_c["sim_revenue"].mean())
                copy_sold = float(res_c["sim_sold"].mean())

            if opponent_mode == "historical":
                score = hist_rev
            elif opponent_mode == "copycat":
                score = copy_rev
            else:
                score = (1.0 - copycat_weight) * hist_rev + copycat_weight * copy_rev

            leaderboard.append({
                **params,
                "score_rev": score,
                "hist_mean_rev": hist_rev,
                "copycat_mean_rev": copy_rev,
                "hist_avg_sold": hist_sold,
                "copycat_avg_sold": copy_sold,
            })
            if i % 6 == 0 or i == total:
                print(f"  Completed {i}/{total} combinations...")
    finally:
        for attr_name, orig_val in saved_attrs.items():
            setattr(duopoly, attr_name, orig_val)
        if hasattr(duopoly, "ENABLE_PYOMO"):
            duopoly.ENABLE_PYOMO = saved_pyomo_flag

    return pd.DataFrame(leaderboard).sort_values("score_rev", ascending=False)


# --- [A] EVALUATE YOUR CURRENT duopoly.py AS-IS (WITH FULL PYOMO ENABLED) ---
eval_slice = raw_df[raw_df["run_id"] == "1111"] if "1111" in raw_df["run_id"].values else raw_df[raw_df["run_id"] == latest_run]

print("\n=== [A] CURRENT duopoly.p() PERFORMANCE (AS-IS IN duopoly.py) ===")
for mode in ("historical", "copycat"):
    base_res = backtest_duopoly_p(eval_slice, opponent_mode=mode, n_competitions=6, train_df=raw_df)
    print(
        f"  Mode={mode:10s} | Mean Sim Rev=${base_res['sim_revenue'].mean():,.2f} | "
        f"Avg Sold={base_res['sim_sold'].mean():.1f}/80 | Max Step={base_res['max_calc_ms'].max():.2f} ms"
    )

# --- [B] GRID-SEARCH BETTER CONSTANTS USING YOUR REAL duopoly.p() ---
PARAM_GRID = {
    "opening_price": [45.0, 55.0, 65.0],
    "normal_undercut": [2.0, 5.0],
    "aggressive_undercut": [6.0, 12.0],
    "harvest_markup": [4.0, 10.0],
    "urgency_threshold": [0.95],
    "harvest_threshold": [0.65],
    "stockout_price": [75.0],
    "min_price_floor": [22.0],
}

tuning_df = tune_duopoly_parameters(
    eval_df=eval_slice,
    param_grid=PARAM_GRID,
    opponent_mode="both",
    copycat_weight=0.5,
    n_competitions=6,
    train_df=raw_df,
    use_pyomo_in_grid=False,  # Fast grid search via duopoly.p(); full Pyomo runs in [A] and [C]
)

print("\n=== [B] TOP 10 PARAMETER SETS (TESTED DIRECTLY VIA duopoly.p) ===")
print(tuning_df.head(10).round(2).to_string(index=False))

best = tuning_df.iloc[0]

# --- [C] VERIFY WINNING CONSTANTS WITH FULL PYOMO ENABLED ---
print("\n=== [C] VERIFYING BEST CONSTANTS IN duopoly.p() (WITH FULL PYOMO ENABLED) ===")
saved_before_verify = {k.upper(): getattr(duopoly, k.upper()) for k in PARAM_GRID.keys()}
try:
    for k in PARAM_GRID.keys():
        setattr(duopoly, k.upper(), float(best[k]))
    for mode in ("historical", "copycat"):
        tuned_res = backtest_duopoly_p(eval_slice, opponent_mode=mode, n_competitions=6, train_df=raw_df)
        print(
            f"  Mode={mode:10s} | Tuned Sim Rev=${tuned_res['sim_revenue'].mean():,.2f} | "
            f"Avg Sold={tuned_res['sim_sold'].mean():.1f}/80 | Max Step={tuned_res['max_calc_ms'].max():.2f} ms"
        )
finally:
    for k, v in saved_before_verify.items():
        setattr(duopoly, k, v)

print("\n=== COPY-PASTE INTO TOP OF duopoly.py ===")
for key in PARAM_GRID.keys():
    print(f"{key.upper()} = {best[key]}")