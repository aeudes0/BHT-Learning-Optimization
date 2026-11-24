from typing import Any, Dict, Optional, Tuple, Union
import os
import pickle
import numpy as np
from scipy.stats import poisson

# === TUNED BASELINE CONSTANTS (USED FOR PERIODS 1–19 COLD START) ===
OPENING_PRICE = 35.0
NORMAL_UNDERCUT = 2.0
AGGRESSIVE_UNDERCUT = 6.0
HARVEST_MARKUP = 4.0
URGENCY_THRESHOLD = 0.95
HARVEST_THRESHOLD = 0.65
STOCKOUT_PRICE = 75.0
MIN_PRICE_FLOOR = 10.0
# ===================================================================

TOTAL_CAPACITY = 80
MAX_PERIODS = 100
MAX_PRICE = 140.0

# Adaptive DP settings (rolling-horizon checkpoints)
DP_CHECKPOINTS = {20, 40, 60, 80}
DP_PRICE_GRID = np.linspace(MIN_PRICE_FLOOR, MAX_PRICE, 131)  # $10 to $140 in $1 steps
ENABLE_PYOMO = False  # Kept for compatibility with simulate_analyze.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ARTIFACT_PATH = os.path.join(BASE_DIR, "model_artifacts.pkl")
FEEDBACK_PATH = "duopoly_feedback.data"

# Precompute static state/demand transition index grids once at import time
_X_GRID = np.arange(TOTAL_CAPACITY + 1)          # Remaining seats: 0..80
_K_GRID = np.arange(25)                          # Truncated single-period demand: 0..24
_SALES_MAT = np.minimum(_X_GRID[:, None], _K_GRID[None, :])       # (81, 25)
_NEXT_X_MAT = np.maximum(0, _X_GRID[:, None] - _K_GRID[None, :])  # (81, 25)


def _load_artifacts() -> Optional[Dict[str, Any]]:
    """Loads pre-trained K-Means scaler/classifier, per-cluster OLS params, and DP tables once."""
    if not os.path.exists(ARTIFACT_PATH):
        return None
    try:
        with open(ARTIFACT_PATH, "rb") as fp:
            return pickle.load(fp)
    except Exception:
        return None


ARTIFACTS = _load_artifacts()


def _log_feedback(line: str, reset: bool = False) -> None:
    """Safely writes competition telemetry to duopoly_feedback.data (never crashes live runs)."""
    try:
        mode = "w" if reset else "a"
        with open(FEEDBACK_PATH, mode, encoding="utf-8") as fp:
            fp.write(line + "\n")
    except Exception:
        pass


def _record_season_summary(
    season: int,
    information_dump: Dict[str, Any],
    prices_hist: Optional[np.ndarray],
    demand_hist: Optional[np.ndarray],
    fallback_price: float = OPENING_PRICE,
) -> None:
    """Logs one summary row per season to duopoly_feedback.data."""
    if information_dump.get("season_logged", False):
        return

    sold_seats = int(information_dump.get("sold_seats", 0))
    if prices_hist is not None and prices_hist.shape[1] > 0:
        p1_all = prices_hist[0, :]
        p2_all = prices_hist[1, :]
        valid_p2 = p2_all[np.isfinite(p2_all) & (p2_all < 500.0)]
        our_avg = float(np.mean(p1_all))
        comp_avg = float(np.mean(valid_p2)) if len(valid_p2) > 0 else -1.0
        if demand_hist is not None and len(demand_hist) > 0:
            n = min(len(p1_all), len(demand_hist))
            est_rev = float(np.sum(p1_all[:n] * demand_hist[:n]))
        else:
            est_rev = 0.0
    else:
        our_avg = float(fallback_price)
        comp_avg = -1.0
        est_rev = 0.0

    _log_feedback(
        f"{season},{int(ARTIFACTS is not None)},{information_dump.get('active_cluster')},"
        f"{sold_seats},{est_rev:.2f},{our_avg:.2f},{comp_avg:.2f}"
    )
    information_dump["season_logged"] = True


def solve_dp_policy_from_t(
    start_t: int,
    beta_0: float,
    beta_1: float,
    beta_2: float,
    comp_price: float,
    T: int = MAX_PERIODS,
    max_seats: int = TOTAL_CAPACITY,
    price_grid: np.ndarray = DP_PRICE_GRID,
) -> np.ndarray:
    """
    Fully vectorized NumPy backward-induction DP solver (< 0.5 ms per call).
    Returns optimal price lookup table P[t, remaining_seats].
    """
    V = np.zeros((T + 2, max_seats + 1), dtype=np.float64)
    P = np.zeros((T + 1, max_seats + 1), dtype=np.float64)

    lambdas = np.maximum(0.001, beta_0 + beta_1 * price_grid + beta_2 * comp_price)
    prob_k = poisson.pmf(_K_GRID[None, :], mu=lambdas[:, None])
    prob_k[:, -1] = 1.0 - prob_k[:, :-1].sum(axis=1)

    # Precompute expected immediate revenue for every (price, remaining_seats) pair
    immediate_rev = price_grid[:, None] * (prob_k @ _SALES_MAT.T)

    for t in range(T, start_t - 1, -1):
        expected_rev = immediate_rev + (prob_k @ V[t + 1, _NEXT_X_MAT].T)
        best_idx = np.argmax(expected_rev, axis=0)
        V[t] = expected_rev[best_idx, _X_GRID]
        P[t] = price_grid[best_idx]

    return P


def _classify_booking_curve(
    p1_hist: np.ndarray,
    p2_hist: np.ndarray,
    d_hist: np.ndarray,
    artifacts: Dict[str, Any],
) -> int:
    """Matches the live season's opening booking curve to the nearest historical cluster in pure NumPy."""
    checkpoint_day = int(artifacts.get("checkpoint_day", 20))
    n = min(checkpoint_day, len(d_hist), p1_hist.shape[0])

    cum_demand_20 = float(np.sum(d_hist[:n])) * (float(checkpoint_day) / float(max(1, n)))
    mean_price_20 = float(np.mean(p1_hist[:n]))

    p2_slice = p2_hist[:n]
    valid_p2 = p2_slice[np.isfinite(p2_slice) & (p2_slice < 500.0)]
    mean_comp_price_20 = float(np.mean(valid_p2)) if len(valid_p2) > 0 else 35.36

    x_vec = np.array([cum_demand_20, mean_price_20, mean_comp_price_20], dtype=np.float64)
    scaler = artifacts["scaler"]
    x_scaled = (x_vec - scaler.mean_) / scaler.scale_

    centers = artifacts["kmeans"].cluster_centers_
    dists = np.sum((centers - x_scaled[None, :]) ** 2, axis=1)
    return int(np.argmin(dists))


def _previous_hybrid_heuristic(
    t: int,
    sold_seats: int,
    remaining_capacity: float,
    remaining_periods: int,
    active_comp_price: float,
    last_price: float,
) -> float:
    """Baseline heuristic used for the cold-start exploration window (Periods 1–19)."""
    if not np.isfinite(active_comp_price) or active_comp_price > 500.0:
        return float(np.clip(OPENING_PRICE, MIN_PRICE_FLOOR, MAX_PRICE))

    target_rate = remaining_capacity / float(max(1, remaining_periods))
    if target_rate > URGENCY_THRESHOLD:
        tier_price = active_comp_price - AGGRESSIVE_UNDERCUT
    elif target_rate < HARVEST_THRESHOLD:
        tier_price = active_comp_price + HARVEST_MARKUP
    else:
        tier_price = active_comp_price - NORMAL_UNDERCUT

    blended = 0.65 * tier_price + 0.35 * last_price
    return float(np.clip(blended, MIN_PRICE_FLOOR, MAX_PRICE))


def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[Optional[float], Any]:
    t = int(selling_period_in_current_season)
    season = int(current_selling_season)

    # 1. Initialize competition-wide persistent state & write feedback header
    if information_dump is None or not isinstance(information_dump, dict) or (season == 1 and t == 1):
        information_dump = {
            "last_price": float(OPENING_PRICE),
            "calibrated_open_price": float(OPENING_PRICE),
            "sold_seats": 0,
            "active_cluster": None,
            "dp_table": None,
            "season_logged": False,
        }
        if season == 1 and t == 1:
            _log_feedback(
                "season,artifacts_loaded,final_cluster,sold_seats,season_rev,our_avg_p,comp_avg_p",
                reset=True,
            )

    # End-of-season cleanup & telemetry hook (t >= 101)
    if t >= 101:
        if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
            information_dump["sold_seats"] = int(np.sum(demand_historical_in_current_season))
        _record_season_summary(
            season=season,
            information_dump=information_dump,
            prices_hist=prices_historical_in_current_season,
            demand_hist=demand_historical_in_current_season,
        )
        information_dump["active_cluster"] = None
        information_dump["dp_table"] = None
        return None, information_dump

    # 2. Start of each season (t == 1): calibrate opening price and reset season DP policy
    if t == 1:
        if season > 1:
            prev_sold = float(information_dump.get("sold_seats", 79))
            open_adj = 1.0 + 0.05 * ((prev_sold - 79.0) / float(TOTAL_CAPACITY))
            base_open = float(information_dump.get("calibrated_open_price", OPENING_PRICE))
            information_dump["calibrated_open_price"] = float(
                np.clip(base_open * open_adj, max(MIN_PRICE_FLOOR, 0.8 * OPENING_PRICE), 1.25 * OPENING_PRICE)
            )
        open_p = float(information_dump.get("calibrated_open_price", OPENING_PRICE))
        information_dump["last_price"] = open_p
        information_dump["sold_seats"] = 0
        information_dump["active_cluster"] = None
        information_dump["dp_table"] = None
        information_dump["season_logged"] = False
        return float(open_p), information_dump

    # 3. Inventory tracking
    sold_seats = (
        int(np.sum(demand_historical_in_current_season))
        if (demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0)
        else 0
    )
    information_dump["sold_seats"] = sold_seats
    remaining_capacity = max(0, TOTAL_CAPACITY - sold_seats)
    remaining_periods = max(1, MAX_PERIODS - t + 1)

    if remaining_capacity <= 0:
        if t == 100:
            _record_season_summary(
                season=season,
                information_dump=information_dump,
                prices_hist=prices_historical_in_current_season,
                demand_hist=demand_historical_in_current_season,
                fallback_price=150.0,
            )
        return 150.0, information_dump

    # 4. Competitor out of stock -> monopoly scarcity pricing
    if not competitor_has_capacity_current_period_in_current_season:
        scarcity_mult = 1.0 + 0.35 * (1.0 - (remaining_capacity / float(TOTAL_CAPACITY)))
        monopoly_price = float(np.clip(STOCKOUT_PRICE * scarcity_mult, MIN_PRICE_FLOOR, 150.0))
        information_dump["last_price"] = monopoly_price
        if t == 100:
            _record_season_summary(
                season=season,
                information_dump=information_dump,
                prices_hist=prices_historical_in_current_season,
                demand_hist=demand_historical_in_current_season,
                fallback_price=monopoly_price,
            )
        return float(monopoly_price), information_dump

    last_price = float(information_dump.get("last_price", OPENING_PRICE))
    active_comp_price = np.nan
    valid_cp = np.array([], dtype=np.float64)

    if prices_historical_in_current_season is not None and prices_historical_in_current_season.shape[1] > 0:
        comp_hist = prices_historical_in_current_season[1, :]
        valid_cp = comp_hist[np.isfinite(comp_hist) & (comp_hist < 500.0)]
        if len(valid_cp) > 0:
            active_comp_price = float(valid_cp[-1])

    heuristic_price = _previous_hybrid_heuristic(
        t=t,
        sold_seats=sold_seats,
        remaining_capacity=remaining_capacity,
        remaining_periods=remaining_periods,
        active_comp_price=active_comp_price,
        last_price=last_price,
    )

    # 5. Cold-start window (Periods 2–19): gather 20-day booking curve with light exploration
    if (
        t < 20
        or ARTIFACTS is None
        or prices_historical_in_current_season is None
        or demand_historical_in_current_season is None
    ):
        explore_offset = 2.0 if (t % 2 == 0) else -2.0
        new_price = float(np.clip(heuristic_price + explore_offset, MIN_PRICE_FLOOR, MAX_PRICE))
        information_dump["last_price"] = new_price
        return float(new_price), information_dump

    # 6. Rolling 20-day checkpoints (t = 20, 40, 60, 80): classify booking curve & refresh DP
    if t in DP_CHECKPOINTS or information_dump.get("dp_table") is None:
        cluster_id = _classify_booking_curve(
            p1_hist=prices_historical_in_current_season[0, :],
            p2_hist=prices_historical_in_current_season[1, :],
            d_hist=demand_historical_in_current_season,
            artifacts=ARTIFACTS,
        )
        params = ARTIFACTS["cluster_models"][cluster_id]
        recent_cp = float(np.mean(valid_cp[-20:])) if len(valid_cp) > 0 else float(params["avg_comp_price"])

        dp_table = solve_dp_policy_from_t(
            start_t=t,
            beta_0=float(params["b0"]),
            beta_1=float(params["b1"]),
            beta_2=float(params["b2"]),
            comp_price=recent_cp,
        )
        information_dump["active_cluster"] = cluster_id
        information_dump["dp_table"] = dp_table

    # 7. Lookup optimal DP price for current (period t, remaining_capacity)
    dp_table = information_dump["dp_table"]
    seat_idx = int(np.clip(remaining_capacity, 1, TOTAL_CAPACITY))
    period_idx = int(np.clip(t, 1, MAX_PERIODS))
    dp_price = float(dp_table[period_idx, seat_idx])

    if dp_price <= 0.0:
        dp_price = heuristic_price

    if np.isfinite(active_comp_price) and active_comp_price < 500.0:
        # Tighter cap (+3) when behind pace; looser cap (+10) when ahead of pace
        pace_ratio = remaining_capacity / float(max(1, remaining_periods))
        max_premium = 3.0 if pace_ratio > 0.75 else 10.0
        dp_price = min(dp_price, active_comp_price + max_premium)

    # Smooth price transitions (prevents harsh jumps at checkpoint boundaries)
    new_price = float(np.clip(dp_price, 0.85 * last_price, 1.18 * last_price))
    final_price = float(np.clip(new_price, MIN_PRICE_FLOOR, MAX_PRICE))
    information_dump["last_price"] = final_price

    if t == 100:
        _record_season_summary(
            season=season,
            information_dump=information_dump,
            prices_hist=prices_historical_in_current_season,
            demand_hist=demand_historical_in_current_season,
            fallback_price=final_price,
        )

    return float(final_price), information_dump