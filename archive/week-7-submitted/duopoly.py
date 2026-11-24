from typing import Any, Dict, Optional, Tuple, Union
import logging
import os
import time
import numpy as np

# Local Windows solver path setup only (never touches paths or env vars on the Linux DPC server)
if os.name == "nt":
    try:
        import idaes  # noqa: F401
    except Exception:
        pass
    for _win_bin in (
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "idaes", "bin"),
        os.path.join(os.environ.get("APPDATA", ""), "idaes", "bin"),
    ):
        if _win_bin and os.path.isdir(_win_bin) and _win_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _win_bin + os.pathsep + os.environ.get("PATH", "")

try:
    from pyomo.environ import (
        ConcreteModel,
        Var,
        RangeSet,
        NonPositiveReals,
        NonNegativeReals,
        NegativeReals,
        Objective,
        Constraint,
        SolverFactory,
        TerminationCondition,
    )
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False

# === PASTE OUTPUT FROM simulate_analyze.py HERE ===
OPENING_PRICE = 65.0
NORMAL_UNDERCUT = 2.0
AGGRESSIVE_UNDERCUT = 6.0
HARVEST_MARKUP = 4.0
URGENCY_THRESHOLD = 0.95
HARVEST_THRESHOLD = 0.65
STOCKOUT_PRICE = 75.0
MIN_PRICE_FLOOR = 22.0
# ==================================================

# Optional embedded target sales curve {period: cumulative_seats} (no external .pkl file needed on DPC)
TARGET_SALES_CURVE: Optional[Dict[int, float]] = None

TOTAL_CAPACITY = 80
MAX_PERIODS = 100
MAX_PRICE = 140.0

# Strict DPC runtime budget controls
ENABLE_PYOMO = True
LOOKBACK_WINDOW = 16
PYOMO_CHECKPOINTS = {18, 46, 74}   # 3 solves/season max (300/comp instead of 8,100)
MAX_PYOMO_TIME_SEC = 8.0           # Hard 8.0s total Pyomo cap per 100-season competition


def _detect_solvers() -> Tuple[Optional[str], Optional[str]]:
    """Detects available LP (glpk on DPC Linux, cbc on Windows) and NLP (ipopt) once at import."""
    if not PYOMO_AVAILABLE:
        return None, None

    pyomo_logger = logging.getLogger("pyomo.opt")
    prev_level = pyomo_logger.level
    pyomo_logger.setLevel(logging.ERROR)
    try:
        lp_name = None
        for candidate in ("glpk", "cbc"):
            opt = SolverFactory(candidate)
            if opt is not None and opt.available(exception_flag=False):
                lp_name = candidate
                break

        nlp_name = None
        opt_nlp = SolverFactory("ipopt")
        if opt_nlp is not None and opt_nlp.available(exception_flag=False):
            nlp_name = "ipopt"

        return lp_name, nlp_name
    except Exception:
        return None, None
    finally:
        pyomo_logger.setLevel(prev_level)


_LP_SOLVER_NAME, _NLP_SOLVER_NAME = _detect_solvers()


def get_target_sales(t: int, target_curve: Dict[int, float]) -> float:
    """Linearly interpolates cumulative target seats sold at period t."""
    days = sorted(target_curve.keys())
    if t in target_curve:
        return float(target_curve[t])
    if len(days) == 1:
        return float(np.clip(target_curve[days[0]] * (t / float(max(1, days[0]))), 0.0, TOTAL_CAPACITY))
    if t <= days[0]:
        return float(target_curve[days[0]] * (t / float(max(1, days[0]))))
    if t >= days[-1]:
        d1, d2 = days[-2], days[-1]
        s1, s2 = float(target_curve[d1]), float(target_curve[d2])
        return float(np.clip(s2 + (s2 - s1) * (t - d2) / float(max(1, d2 - d1)), 0.0, TOTAL_CAPACITY))
    for i in range(len(days) - 1):
        d1, d2 = days[i], days[i + 1]
        if d1 <= t <= d2:
            s1, s2 = float(target_curve[d1]), float(target_curve[d2])
            return float(s1 + (s2 - s1) * (t - d1) / float(d2 - d1))
    return float(TOTAL_CAPACITY * (t / float(MAX_PERIODS)))


def estimate_demand_pyomo(
    d: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    window: int = LOOKBACK_WINDOW,
) -> Optional[Tuple[float, float]]:
    """Crash-proof, time-bounded Bertsimas & Perakis (2001) 4-Step LP + NLP estimator."""
    if not ENABLE_PYOMO or _LP_SOLVER_NAME is None or _NLP_SOLVER_NAME is None:
        return None

    try:
        valid_mask = np.isfinite(p2) & (p2 < 500.0) & np.isfinite(p1) & np.isfinite(d)
        d_valid = d[valid_mask][-window:]
        p1_valid = p1[valid_mask][-window:]
        p2_valid = p2[valid_mask][-window:]

        MaxT = len(d_valid)
        if MaxT < 6 or np.std(p1_valid) < 0.25:
            return None

        # STEP 1: Firm 1 Time-Varying Linear Demand Estimation via LP (glpk / cbc)
        model1 = ConcreteModel("Demand_Estim_1")
        delta1 = 0.05

        T = RangeSet(1, MaxT, 1)
        K1 = RangeSet(1, 1, 1)

        model1.beta0 = Var(K1, T, within=NonNegativeReals)
        model1.beta1 = Var(K1, T, within=NonPositiveReals)
        model1.beta2 = Var(K1, T, within=NonNegativeReals)
        model1.deltaTau = Var(T, within=NonNegativeReals)

        model1.obj = Objective(rule=lambda m: sum(m.deltaTau[t] for t in T))

        model1.first = Constraint(
            T,
            rule=lambda m, t: d_valid[t - 1]
            - (m.beta0[1, t] + m.beta1[1, t] * p1_valid[t - 1] + m.beta2[1, t] * p2_valid[t - 1])
            <= m.deltaTau[t],
        )
        model1.second = Constraint(
            T,
            rule=lambda m, t: -d_valid[t - 1]
            + (m.beta0[1, t] + m.beta1[1, t] * p1_valid[t - 1] + m.beta2[1, t] * p2_valid[t - 1])
            <= m.deltaTau[t],
        )
        model1.third0 = Constraint(
            T, rule=lambda m, t: (m.beta0[1, t] - m.beta0[1, t + 1] <= delta1) if t < MaxT else Constraint.Feasible
        )
        model1.fourth0 = Constraint(
            T, rule=lambda m, t: (-m.beta0[1, t] + m.beta0[1, t + 1] <= delta1) if t < MaxT else Constraint.Feasible
        )
        model1.third1 = Constraint(
            T, rule=lambda m, t: (m.beta1[1, t] - m.beta1[1, t + 1] <= delta1) if t < MaxT else Constraint.Feasible
        )
        model1.fourth1 = Constraint(
            T, rule=lambda m, t: (-m.beta1[1, t] + m.beta1[1, t + 1] <= delta1) if t < MaxT else Constraint.Feasible
        )
        model1.third2 = Constraint(
            T, rule=lambda m, t: (m.beta2[1, t] - m.beta2[1, t + 1] <= delta1) if t < MaxT else Constraint.Feasible
        )
        model1.fourth2 = Constraint(
            T, rule=lambda m, t: (-m.beta2[1, t] + m.beta2[1, t + 1] <= delta1) if t < MaxT else Constraint.Feasible
        )

        opt1 = SolverFactory(_LP_SOLVER_NAME)
        if _LP_SOLVER_NAME == "glpk":
            opt1.options["tmlim"] = 1
        elif _LP_SOLVER_NAME == "cbc":
            opt1.options["seconds"] = 1.0

        res1 = opt1.solve(model1, tee=False)
        if res1.solver.termination_condition != TerminationCondition.optimal:
            return None

        N = min(10, MaxT)
        b10_vals = [model1.beta0[1, t].value for t in T][-N:]
        b11_vals = [model1.beta1[1, t].value for t in T][-N:]
        b12_vals = [model1.beta2[1, t].value for t in T][-N:]
        if any(v is None for v in b10_vals + b11_vals + b12_vals):
            return None

        beta10t = float(np.mean(b10_vals))
        beta11t = float(np.mean(b11_vals))
        beta12t = float(np.mean(b12_vals))

        # STEP 3: Competitor Inverse Demand Estimation via NLP (ipopt)
        model2 = ConcreteModel("Demand_Estim_2")
        delta2 = 0.01

        K2 = RangeSet(2, 2, 1)
        model2.beta0 = Var(K2, T, within=NonNegativeReals, bounds=(0.0, 1000.0), initialize=10.0)
        model2.beta1 = Var(K2, T, within=NonNegativeReals, bounds=(0.0, 100.0), initialize=0.1)
        model2.beta2 = Var(K2, T, within=NegativeReals, bounds=(-100.0, -0.05), initialize=-0.5)

        model2.obj = Objective(
            rule=lambda m: sum(
                (p2_valid[t - 1] + (m.beta0[2, t] + m.beta1[2, t] * p1_valid[t - 1]) / (2.0 * m.beta2[2, t])) ** 2
                for t in T
            )
        )
        model2.third0 = Constraint(
            T, rule=lambda m, t: ((m.beta0[2, t] - m.beta0[2, t + 1]) ** 2 <= delta2) if t < MaxT else Constraint.Feasible
        )
        model2.third1 = Constraint(
            T, rule=lambda m, t: ((m.beta1[2, t] - m.beta1[2, t + 1]) ** 2 <= delta2) if t < MaxT else Constraint.Feasible
        )
        model2.third2 = Constraint(
            T, rule=lambda m, t: ((m.beta2[2, t] - m.beta2[2, t + 1]) ** 2 <= delta2) if t < MaxT else Constraint.Feasible
        )

        opt2 = SolverFactory(_NLP_SOLVER_NAME)
        opt2.options["tol"] = 1e-4
        opt2.options["max_iter"] = 80
        opt2.options["max_cpu_time"] = 0.5
        opt2.options["print_level"] = 0
        res2 = opt2.solve(model2, tee=False)
        if res2.solver.termination_condition not in (
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
        ):
            return None

        b20_vals = [model2.beta0[2, t].value for t in T][-N:]
        b21_vals = [model2.beta1[2, t].value for t in T][-N:]
        b22_vals = [model2.beta2[2, t].value for t in T][-N:]
        if any(v is None for v in b20_vals + b21_vals + b22_vals):
            return None

        beta20t = float(np.mean(b20_vals))
        beta21t = float(np.mean(b21_vals))
        beta22t = float(np.mean(b22_vals))

        if abs(beta22t) < 1e-6:
            return None

        denom2 = -2.0 * beta22t
        A = beta10t + beta12t * (beta20t / denom2)
        B = beta11t + beta12t * (beta21t / denom2)

        if not (np.isfinite(A) and np.isfinite(B)) or B >= -1e-5 or A <= 0.05:
            return None

        return float(A), float(B)
    except Exception:
        return None


def _solve_step4_closed_form(
    A: float,
    B: float,
    remaining_capacity: float,
    remaining_periods: int,
    min_price: float = MIN_PRICE_FLOOR,
    max_price: float = MAX_PRICE,
) -> float:
    """Exact O(1) analytical Step 4 revenue maximization + Theorem 2 capacity floor."""
    opt_price = float(np.clip(-A / (2.0 * B), min_price, max_price))

    expected_period_demand = A + B * opt_price
    if remaining_capacity > 0 and expected_period_demand * remaining_periods > remaining_capacity:
        target_demand = remaining_capacity / float(remaining_periods)
        cap_price = (target_demand - A) / B
        opt_price = max(opt_price, cap_price)

    return float(np.clip(opt_price, min_price, max_price))


def _previous_hybrid_heuristic(
    t: int,
    sold_seats: int,
    remaining_capacity: float,
    remaining_periods: int,
    active_comp_price: float,
    last_price: float,
) -> float:
    """Combines your 8 tuned constants with optional embedded TARGET_SALES_CURVE."""
    if not np.isfinite(active_comp_price) or active_comp_price > 500.0:
        return float(np.clip(OPENING_PRICE, MIN_PRICE_FLOOR, MAX_PRICE))

    target_rate = remaining_capacity / float(max(1, remaining_periods))
    if target_rate > URGENCY_THRESHOLD:
        tier_price = active_comp_price - AGGRESSIVE_UNDERCUT
    elif target_rate < HARVEST_THRESHOLD:
        tier_price = active_comp_price + HARVEST_MARKUP
    else:
        tier_price = active_comp_price - NORMAL_UNDERCUT

    if TARGET_SALES_CURVE:
        target_sales = get_target_sales(t, TARGET_SALES_CURVE)
        norm_delta = float(np.clip((sold_seats - target_sales) / max(10.0, target_sales), -0.25, 0.30))
        tier_price *= 1.0 + 0.5 * norm_delta

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

    # 1. Initialize competition-wide persistent state
    if information_dump is None or not isinstance(information_dump, dict) or (season == 1 and t == 1):
        information_dump = {
            "last_price": float(OPENING_PRICE),
            "calibrated_open_price": float(OPENING_PRICE),
            "cached_AB": None,
            "pyomo_time_used": 0.0,
            "sold_seats": 0,
        }

    # Optional post-season hook (if called by local backtester at t >= 101)
    if t >= 101:
        if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
            information_dump["sold_seats"] = int(np.sum(demand_historical_in_current_season))
        information_dump["cached_AB"] = None
        return None, information_dump

    # 2. Start of each season (t == 1): calibrate opening price from previous season's final load factor
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
        return 150.0, information_dump

    # 4. Competitor out of stock -> monopoly scarcity pricing
    if not competitor_has_capacity_current_period_in_current_season:
        scarcity_mult = 1.0 + 0.35 * (1.0 - (remaining_capacity / float(TOTAL_CAPACITY)))
        monopoly_price = float(np.clip(STOCKOUT_PRICE * scarcity_mult, MIN_PRICE_FLOOR, 150.0))
        information_dump["last_price"] = monopoly_price
        return float(monopoly_price), information_dump

    last_price = float(information_dump.get("last_price", OPENING_PRICE))
    active_comp_price = np.nan
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

    # 5. Early burn-in exploration (t = 2..17) to create price variance before first Pyomo solve at t=18
    if t < 18 or prices_historical_in_current_season is None or demand_historical_in_current_season is None:
        explore_offset = 2.0 if (t % 2 == 0) else -2.0
        new_price = float(np.clip(heuristic_price + explore_offset, MIN_PRICE_FLOOR, MAX_PRICE))
        information_dump["last_price"] = new_price
        return float(new_price), information_dump

    # 6. Checkpoint Pyomo LP+NLP solve (max 3x/season & hard-capped at MAX_PYOMO_TIME_SEC = 8.0s/competition)
    if (
        ENABLE_PYOMO
        and t in PYOMO_CHECKPOINTS
        and information_dump.get("pyomo_time_used", 0.0) < MAX_PYOMO_TIME_SEC
    ):
        t0 = time.perf_counter()
        ab_est = estimate_demand_pyomo(
            d=demand_historical_in_current_season,
            p1=prices_historical_in_current_season[0, :],
            p2=prices_historical_in_current_season[1, :],
        )
        information_dump["pyomo_time_used"] = float(information_dump.get("pyomo_time_used", 0.0)) + (
            time.perf_counter() - t0
        )
        if ab_est is not None:
            information_dump["cached_AB"] = ab_est

    # 7. Exact O(1) Step 4 optimal price from cached (A, B) (warm-starts from prior season until t=18 updates it)
    cached_ab = information_dump.get("cached_AB")
    if cached_ab is not None:
        A, B = cached_ab
        pyomo_opt = _solve_step4_closed_form(
            A=A,
            B=B,
            remaining_capacity=remaining_capacity,
            remaining_periods=remaining_periods,
            min_price=MIN_PRICE_FLOOR,
            max_price=MAX_PRICE,
        )
        new_price = 0.25 * pyomo_opt + 0.75 * heuristic_price
    else:
        new_price = heuristic_price

    new_price = float(np.clip(new_price, 0.88 * last_price, 1.15 * last_price))
    final_price = float(np.clip(new_price, MIN_PRICE_FLOOR, MAX_PRICE))
    information_dump["last_price"] = final_price

    return float(final_price), information_dump