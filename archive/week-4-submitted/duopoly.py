from typing import Any, Optional, Tuple, Union
import numpy as np
import os
import pickle

TOTAL_CAPACITY = 80
MAX_PERIODS = 100

def load_target_curve():
    """Loads target sales curve dictionary from pkl if present, else generates linear fallback."""
    if os.path.exists('target_sales_curve.pkl'):
        try:
            with open('target_sales_curve.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return {t: int(TOTAL_CAPACITY * (t / MAX_PERIODS)) for t in np.arange(1, 101, 5)}

def get_target_sales(t: int, target_curve: dict) -> float:
    """Safely extracts target sold seats at period t using linear interpolation/extrapolation."""
    days = sorted(target_curve.keys())
    
    def extract_seats(val):
        if isinstance(val, dict):
            return float(val.get('sold_seats', 0.0))
        return float(val)

    if t in target_curve:
        return extract_seats(target_curve[t])
    
    # Boundary extrapolation
    if t <= days[0]:
        s0 = extract_seats(target_curve[days[0]])
        return float(s0 * (t / max(1, days[0])))
    if t >= days[-1]:
        d1, d2 = days[-2], days[-1]
        s1, s2 = extract_seats(target_curve[d1]), extract_seats(target_curve[d2])
        extrapolated = s2 + (s2 - s1) * (t - d2) / max(1, d2 - d1)
        return float(np.clip(extrapolated, 0.0, TOTAL_CAPACITY))

    # Interpolation between closest knots
    for i in range(len(days) - 1):
        d1, d2 = days[i], days[i + 1]
        if d1 <= t <= d2:
            s1, s2 = extract_seats(target_curve[d1]), extract_seats(target_curve[d2])
            return float(s1 + (s2 - s1) * (t - d1) / (d2 - d1))
            
    return float(TOTAL_CAPACITY * (t / MAX_PERIODS))

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    t = selling_period_in_current_season
    
    # 1. State Initialization
    if t == 1 or information_dump is None or not isinstance(information_dump, dict):
        information_dump = {
            'sold_seats': 0,
            'last_price': 50.0,
            'target_curve': load_target_curve(),
            'observation_history': []
        }
        
    target_curve = information_dump.get('target_curve', load_target_curve())
    
    # 2. Cumulative Sales Synchronization
    if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
        sold_seats = int(np.sum(demand_historical_in_current_season))
    else:
        sold_seats = 0
    information_dump['sold_seats'] = sold_seats
    remaining_capacity = max(0, TOTAL_CAPACITY - sold_seats)
    
    # Stockout protection
    if remaining_capacity <= 0:
        return 150.0, information_dump

    # 3. Monopoly Mode (Competitor out of stock)
    if not competitor_has_capacity_current_period_in_current_season:
        scarcity_factor = 1.0 + (1.0 - (remaining_capacity / TOTAL_CAPACITY))
        monopoly_price = (55.0 + 40.0 * (t / MAX_PERIODS)) * scarcity_factor
        price = float(np.clip(monopoly_price, 35.0, 150.0))
        information_dump['last_price'] = price
        return price, information_dump

    if t == 1:
        base_price = 50.0
        information_dump['last_price'] = base_price
        return base_price, information_dump

    # 4. Period Update Frequency Schedule
    should_update = False
    if t < 75:
        if (t - 1) % 3 == 0:
            should_update = True
    elif t < 90:
        if t % 2 == 1:
            should_update = True
    else:
        should_update = True

    if not should_update:
        return information_dump.get('last_price', 50.0), information_dump

    # 5. Pace Target & Reference Price Anchoring
    target_sales = get_target_sales(t, target_curve)
    reference_price = information_dump.get('last_price', 50.0)
    
    if prices_historical_in_current_season is not None and prices_historical_in_current_season.shape[1] > 0:
        comp_price = prices_historical_in_current_season[1, -1]
        pace_ratio = sold_seats / max(1.0, target_sales)
        
        if comp_price >= 20.0 and pace_ratio >= 0.8: 
            reference_price = 0.8 * reference_price + 0.2 * comp_price

    # 6. Smooth Pace Deficit / Surplus Feedback Adjustment
    delta = (sold_seats - target_sales) / max(1.0, target_sales)
    pricing_multiplier = 1.2 if delta > 0 else 0.8
    heuristic_price = reference_price * (1.0 + pricing_multiplier * delta)

    new_price = heuristic_price

    # 7. ARIMA-Inspired ARX(1) Demand Model Estimation
    # (AutoRegressive order 1 with Exogenous variables: Own Price and Comp Price)
    if t > 6 and prices_historical_in_current_season is not None and demand_historical_in_current_season is not None:
        try:
            p_own_hist = prices_historical_in_current_season[0, :]
            p_comp_hist = prices_historical_in_current_season[1, :]
            d_hist = demand_historical_in_current_season[:]
            
            # Require enough variation and history for AR(1) shift
            if np.std(p_own_hist) > 0.5 and len(d_hist) > 2:
                # Shift arrays to align current period (t) with lag (t-1)
                p_own_curr = p_own_hist[1:]
                p_comp_curr = p_comp_hist[1:]
                d_curr = d_hist[1:]
                d_lag1 = d_hist[:-1]  # The Autoregressive (AR1) feature
                
                N_obs = len(d_curr)
                # Feature Matrix: [Intercept, Our Price, Comp Price, Lagged Demand]
                X = np.column_stack((np.ones(N_obs), p_own_curr, p_comp_curr, d_lag1))
                y = d_curr
                
                # Fit ARX model
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                b0, b1, b2, phi1 = beta
                
                latest_comp_price = p_comp_hist[-1]
                last_demand = d_hist[-1]
                
                # Ensure the model passes economic logic: price coefficient (b1) must be negative
                # base_demand now includes the autoregressive momentum (phi1 * last_demand)
                base_demand_est = b0 + b2 * latest_comp_price + phi1 * last_demand
                
                if b1 < -0.01 and base_demand_est > 0:
                    # Maximize Revenue = Price * (Base_Demand_Est + b1 * Price)
                    # Derivative of Rev w.r.t Price -> Base_Demand_Est + 2 * b1 * Price = 0
                    optimal_price = -base_demand_est / (2.0 * b1)
                    optimal_price = np.clip(optimal_price, 20.0, 130.0)
                    
                    # Blend 40% ARX optimization with 60% Target Heuristic
                    new_price = 0.6 * heuristic_price + 0.4 * optimal_price
        except Exception:
            pass

    # 8. Inventory Scarcity & End-of-Season Clearance Adjustment
    time_ratio = t / MAX_PERIODS
    cap_ratio = remaining_capacity / TOTAL_CAPACITY
    
    expected_remaining_ratio = 1.0 - time_ratio
    if cap_ratio < expected_remaining_ratio * 0.7:
        new_price *= 1.25
    elif cap_ratio > expected_remaining_ratio * 1.3 and t > 50:
        new_price *= 0.85

    # Final price bounding
    final_price = float(np.clip(new_price, 18.0, 140.0))
    information_dump['last_price'] = final_price
    
    return final_price, information_dump