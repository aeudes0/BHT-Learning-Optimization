# %% [1] IMPORTS & MULTI-FILE LOADER (READS ./result SUBFOLDER)
import glob
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "results")
CAPACITY_PER_SEASON = 80
MAX_RUNS_ON_LEARNING_PLOT = 4  # Caps lines on learning plot so it stays readable


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
    # Exclude sentinel 999 prices when computing active competitor average price
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
        error_count=("errors", lambda s: s.notna().sum())
    )
    seasons["load_factor"] = seasons["total_demand"] / CAPACITY_PER_SEASON
    seasons["realized_yield"] = np.where(
        seasons["total_demand"] > 0,
        seasons["total_revenue"] / seasons["total_demand"],
        0.0
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
    errors=("error_count", "sum")
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
        comp_stockout_periods=("comp_stockout_periods", "mean")
    )
    .sort_values("mean_rev", ascending=False)
)

print(f"\n=== OPPONENT BREAKDOWN (LATEST RUN: {latest_run}) ===")
print(opp_breakdown.round(2).to_string())

# %% [3] CLEAN, UNCLUTTERED DIAGNOSTIC CHARTS (2x2 GRID)
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Chart 1: Average Revenue per Run (Scales cleanly across all 10-15+ files)
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
        fontsize=9
    )

# Chart 2: Season-over-Season Learning Curve (Capped at latest 4 runs)
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

# Chart 3: Within-Season Inventory Burn (Latest Run Only)
ax3 = axes[1, 0]
latest_raw = raw_df[raw_df["run_id"] == latest_run].copy()
latest_raw["cum_demand"] = latest_raw.groupby(["competition_id", "selling_season"])["demand"].cumsum()
burn = latest_raw.groupby("selling_period")["cum_demand"].agg(
    mean="mean",
    p25=lambda s: s.quantile(0.25),
    p75=lambda s: s.quantile(0.75)
)

ax3.plot(burn.index, burn["mean"], label="Mean Cumulative Sales", color="#1b4965", linewidth=2.5)
ax3.fill_between(burn.index, burn["p25"], burn["p75"], color="#62b6cb", alpha=0.25, label="25th–75th Percentile")
ax3.plot([1, 100], [0.8, 80], "--", color="#d9534f", linewidth=1.5, label="Ideal Linear Pace (0.8/period)")
ax3.set_title(f"Intraseason Inventory Burn (Run {latest_run})", fontsize=12, fontweight="bold")
ax3.set_xlabel("Selling Period (1–100)")
ax3.set_ylabel("Cumulative Units Sold")
ax3.set_ylim(0, 85)
ax3.legend(frameon=True)

# Chart 4: Intraseason Price Trajectory (Latest Run Only)
ax4 = axes[1, 1]
price_traj = latest_raw.groupby("selling_period").agg(
    our_price=("price", "mean"),
    comp_price=("price_competitor", lambda s: s[s < 900].mean())
)
ax4.plot(price_traj.index, price_traj["our_price"], label="Our Avg Price", color="#2a9d8f", linewidth=2.5)
ax4.plot(price_traj.index, price_traj["comp_price"], label="Competitor Avg Price", color="#e76f51", linestyle="--", linewidth=2.0)
ax4.set_title(f"Intraseason Price Curve (Run {latest_run})", fontsize=12, fontweight="bold")
ax4.set_xlabel("Selling Period (1–100)")
ax4.set_ylabel("Price ($)")
ax4.legend(frameon=True)

plt.tight_layout()
plt.show()

# %% [4] FAST EMPIRICAL DEMAND MODEL & DUAL-MODE SIMULATOR (HISTORICAL OR COPYCAT)
class EmpiricalDemandEnvironment:
    """
    Precomputes a dense 4D NumPy array for O(1) scalar demand rate lookups
    without calling np.digitize or pandas inside the inner simulation loop.
    """
    def __init__(self, df: pd.DataFrame):
        valid = df.copy()
        valid["price_bin"] = pd.cut(valid["price"], bins=[-1e9, 20, 30, 40, 50, 65, 85, 1e9], labels=False)
        valid["diff_bin"] = pd.cut(
            np.where(valid["competitor_has_capacity"], valid["price"] - valid["price_competitor"], -999.0),
            bins=[-1e9, -100, -15, -5, 5, 15, 1e9],
            labels=False
        )
        valid["period_bin"] = pd.cut(valid["selling_period"], bins=[-1e9, 25, 50, 75, 1e9], labels=False)

        active_fallback = float(valid[valid["competitor_has_capacity"]]["demand"].mean())
        # Shape: [comp_has_cap (0 or 1), period_bin (0..3), price_bin (0..6), diff_bin (0..5)]
        self.table = np.full((2, 4, 7, 6), active_fallback, dtype=np.float64)

        # Smooth exponential fallback when competitor is stocked out (cap == 0)
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


def backtest_strategy(
    eval_df: pd.DataFrame,
    strategy_fn,
    opponent_mode: str = "historical",
    n_competitions: int = 6,
    seed: int = 42,
    copycat_initial_price: float = None,
    train_df: pd.DataFrame = None,
    env: EmpiricalDemandEnvironment = None
) -> pd.DataFrame:
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

    comp_arr = sub["competition_id"].to_numpy()
    opp_arr = sub["competitor_id"].to_numpy()
    season_arr = sub["selling_season"].to_numpy()
    period_arr = sub["selling_period"].to_numpy()
    comp_price_arr = sub["price_competitor"].to_numpy(dtype=np.float64)
    comp_cap_arr = sub["competitor_has_capacity"].to_numpy(dtype=bool)
    rev_arr = sub["revenue"].to_numpy(dtype=np.float64)

    n_rows = len(sub)
    results = []
    idx = 0

    while idx < n_rows:
        end = idx + 100
        our_inv = CAPACITY_PER_SEASON
        comp_inv = CAPACITY_PER_SEASON
        our_rev = 0.0
        comp_rev = 0.0
        last_comp_price_seen = np.nan
        our_prev_price = np.nan
        our_history = {"prices": [], "comp_prices": [], "demands": []}

        for step in range(idx, end):
            period = int(period_arr[step])
            logged_cp = comp_price_arr[step]

            if opponent_mode == "historical":
                comp_has_cap = bool(comp_cap_arr[step])
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
                        comp_price = 40.0
                else:
                    comp_price = our_prev_price

            if our_inv <= 0:
                our_price = 150.0
                our_demand = 0
            else:
                our_price = float(strategy_fn(
                    period=period,
                    remaining_inventory=our_inv,
                    last_comp_price=last_comp_price_seen,
                    comp_has_capacity=comp_has_cap,
                    history=our_history
                ))
                if our_price < 1.0:
                    our_price = 1.0
                elif our_price > 150.0:
                    our_price = 150.0

                lam_our = env.expected_demand(our_price, comp_price, comp_has_cap, period)
                d = int(poisson(lam_our))
                our_demand = our_inv if d > our_inv else d

            if opponent_mode == "copycat" and comp_has_cap:
                our_has_cap = our_inv > 0
                lam_comp = env.expected_demand(comp_price, our_price, our_has_cap, period)
                cd = int(poisson(lam_comp))
                comp_demand = comp_inv if cd > comp_inv else cd
                comp_inv -= comp_demand
                comp_rev += comp_price * comp_demand

            our_inv -= our_demand
            our_rev += our_price * our_demand
            our_prev_price = our_price
            last_comp_price_seen = comp_price

            our_history["prices"].append(our_price)
            our_history["comp_prices"].append(comp_price)
            our_history["demands"].append(our_demand)

        results.append({
            "competition_id": comp_arr[idx],
            "competitor_id": opp_arr[idx],
            "selling_season": int(season_arr[idx]),
            "opponent_mode": opponent_mode,
            "sim_revenue": our_rev,
            "sim_sold": CAPACITY_PER_SEASON - our_inv,
            "comp_sim_revenue": comp_rev if opponent_mode == "copycat" else np.nan,
            "comp_sim_sold": (CAPACITY_PER_SEASON - comp_inv) if opponent_mode == "copycat" else np.nan,
            "actual_logged_revenue": float(rev_arr[idx:end].sum())
        })
        idx = end

    return pd.DataFrame(results)


# %% [5] PARAMETERIZED STRATEGY & DUAL-MODE PARAMETER TUNER
def parameterized_strategy(period, remaining_inventory, last_comp_price, comp_has_capacity, history, params):
    periods_left = 101 - period
    target_rate = remaining_inventory / (periods_left if periods_left > 1 else 1)

    if not comp_has_capacity:
        return params["stockout_price"]

    if np.isnan(last_comp_price) or last_comp_price > 500.0:
        return params["opening_price"]

    if target_rate > params["urgency_threshold"]:
        raw_price = last_comp_price - params["aggressive_undercut"]
    elif target_rate < params["harvest_threshold"]:
        raw_price = last_comp_price + params["harvest_markup"]
    else:
        raw_price = last_comp_price - params["normal_undercut"]

    if raw_price < params["min_price_floor"]:
        return params["min_price_floor"]
    if raw_price > 140.0:
        return 140.0
    return raw_price


def tune_parameters(
    eval_df: pd.DataFrame,
    param_grid: dict,
    opponent_mode: str = "both",
    copycat_weight: float = 0.5,
    n_competitions: int = 6,
    seed: int = 42,
    train_df: pd.DataFrame = None
) -> pd.DataFrame:
    if opponent_mode not in ("historical", "copycat", "both"):
        raise ValueError("opponent_mode must be 'historical', 'copycat', or 'both'")

    print("Building empirical demand table once from all historical logs...")
    shared_env = EmpiricalDemandEnvironment(train_df if train_df is not None else eval_df)

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"Tuning {total} combinations | mode='{opponent_mode}' | competitions={n_competitions}")

    leaderboard = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        strat = lambda **kw: parameterized_strategy(**kw, params=params)

        hist_rev, hist_sold = np.nan, np.nan
        copy_rev, copy_sold = np.nan, np.nan

        if opponent_mode in ("historical", "both"):
            res_h = backtest_strategy(
                eval_df, strat, opponent_mode="historical",
                n_competitions=n_competitions, seed=seed, env=shared_env
            )
            hist_rev = float(res_h["sim_revenue"].mean())
            hist_sold = float(res_h["sim_sold"].mean())

        if opponent_mode in ("copycat", "both"):
            res_c = backtest_strategy(
                eval_df, strat, opponent_mode="copycat",
                n_competitions=n_competitions, seed=seed, env=shared_env
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
            "copycat_avg_sold": copy_sold
        })
        if i % 6 == 0 or i == total:
            print(f"  Completed {i}/{total} combinations...")

    return pd.DataFrame(leaderboard).sort_values("score_rev", ascending=False)


# --- RUN EVALUATION & PARAMETER TUNING ---
# Evaluate against your latest dated run ("1111") or `latest_run`
eval_slice = raw_df[raw_df["run_id"] == "1111"] if "1111" in raw_df["run_id"].values else raw_df[raw_df["run_id"] == latest_run]

PARAM_GRID = {
    "opening_price": [45.0, 55.0, 65.0],
    "normal_undercut": [2.0, 5.0],
    "aggressive_undercut": [6.0, 12.0],
    "harvest_markup": [4.0, 10.0],
    "urgency_threshold": [0.95],
    "harvest_threshold": [0.65],
    "stockout_price": [75.0],
    "min_price_floor": [22.0]
}

tuning_df = tune_parameters(
    eval_df=eval_slice,
    param_grid=PARAM_GRID,
    opponent_mode="both",
    copycat_weight=0.5,
    n_competitions=6,
    train_df=raw_df
)

print("\n=== TOP 10 PARAMETER SETS (BLENDED HISTORICAL + COPYCAT SCORE) ===")
print(tuning_df.head(10).round(2).to_string(index=False))

best = tuning_df.iloc[0]
print("\n=== COPY-PASTE INTO TOP OF duopoly.py ===")
for key in PARAM_GRID.keys():
    print(f"{key.upper()} = {best[key]}")