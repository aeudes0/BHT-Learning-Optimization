from typing import Any, Optional, Tuple, Union
import numpy as np
import os
import pickle

TOTAL_CAPACITY = 80
MAX_PERIODS = 100

def load_target_curve():
    if os.path.exists('target_sales_curve.pkl'):
        try:
            with open('target_sales_curve.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return {t: int(TOTAL_CAPACITY * (t / MAX_PERIODS)) for t in np.arange(1, 101, 5)}

def get_target_sales(t: int, target_curve: dict) -> float:
    days = sorted(target_curve.keys())
    if t in target_curve:
        return float(target_curve[t]['sold_seats'] if isinstance(target_curve[t], dict) else target_curve[t])
    for i in range(len(days) - 1):
        d1, d2 = days[i], days[i + 1]
        if d1 <= t <= d2:
            s1 = target_curve[d1]['sold_seats'] if isinstance(target_curve[d1], dict) else target_curve[d1]
            s2 = target_curve[d2]['sold_seats'] if isinstance(target_curve[d2], dict) else target_curve[d2]
            return float(s1 + (s2 - s1) * (t - d1) / (d2 - d1))
    return float(TOTAL_CAPACITY)

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    t = selling_period_in_current_season
    
    # 1. Initialize information_dump if it's empty or Period 1
    if t == 1 or information_dump is None or not isinstance(information_dump, dict):
        target_curve = load_target_curve()
        information_dump = {
            'sold_seats': 0,
            'last_price': 65.0,
            'target_curve': target_curve,
            'observation_history': []  # <--- CRITICAL: Initialize the history list
        }
        
    # 2. SAFETY CHECK: If an old information_dump was loaded without the history key, add it
    if 'observation_history' not in information_dump:
        information_dump['observation_history'] = []

    target_curve = information_dump.get('target_curve', load_target_curve())
    
    # Update sold seats and record history from the previous period
    if t > 1 and demand_historical_in_current_season is not None and prices_historical_in_current_season is not None:
        last_demand = demand_historical_in_current_season[-1]
        information_dump['sold_seats'] += int(last_demand)
        
        # Capture last period's prices for OLS history
        try:
            last_our_price = prices_historical_in_current_season[0, -1]
            last_comp_price = prices_historical_in_current_season[1, -1]
            information_dump['observation_history'].append((last_our_price, last_comp_price, last_demand))
        except Exception:
            pass # Failsafe if array shapes are weird early on
            
    sold_seats = information_dump['sold_seats']
    remaining_capacity = max(0, TOTAL_CAPACITY - sold_seats)
    
    # Stockout protection: if out of seats, price extremely high
    if remaining_capacity <= 0:
        return 150.0, information_dump

    # Monopoly Mode
    if not competitor_has_capacity_current_period_in_current_season:
        monopoly_price = 80.0 + 60.0 * (t / MAX_PERIODS)
        price = float(np.clip(monopoly_price, 40.0, 150.0))
        information_dump['last_price'] = price
        return price, information_dump

    if t == 1:
        base_price = 65.0
        information_dump['last_price'] = base_price
        return base_price, information_dump

    # Check your update frequency schedule:
    # - Days 1 to 74: every 3 days
    # - Days 75 to 89: every 2 days
    # - Days 90 to 100: every day
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
        return information_dump.get('last_price', 65.0), information_dump

    target_sales = get_target_sales(t, target_curve)
    reference_price = information_dump.get('last_price', 65.0)
    
    if prices_historical_in_current_season is not None:
        comp_price = prices_historical_in_current_season[1, -1]
        
        # Only anchor to the competitor if their price is somewhat reasonable 
        # and we aren't desperately behind on sales.
        delta = (sold_seats - target_sales) / max(1.0, target_sales)
        
        if comp_price >= 25.0 and delta > -0.2: 
            # We are doing okay (not more than 20% behind target), 
            # so we can afford to follow the competitor's lead slightly.
            # Weaken the pull to 20% instead of 50%
            reference_price = 0.8 * reference_price + 0.2 * comp_price
        # If delta <= -0.2 (we are severely behind), we completely IGNORE 
        # the competitor's price and focus solely on our own demand curve.
    # Calculate percentage deviation from target curve
    delta = (sold_seats - target_sales) / max(1.0, target_sales)
    
    # If delta > 0 (selling too fast), increase price more aggressively (multiplier 2)
    # If delta < 0 (selling too slow), lower price to pick up sales (multiplier 0.5)
    pricing_multiplier = 2 if delta > 0 else 0.5
    heuristic_price = reference_price * (1.0 + pricing_multiplier * delta)

    # Set our base intended price to the heuristic
    new_price = heuristic_price

    # Calculate percentage deviation from target curve
    delta = (sold_seats - target_sales) / max(1.0, target_sales)
    
    # If delta > 0 (selling too fast), increase price more aggressively (multiplier 2)
    # If delta < 0 (selling too slow), lower price to pick up sales (multiplier 0.5)
    pricing_multiplier = 2 if delta > 0 else 0.5
    heuristic_price = reference_price * (1.0 + pricing_multiplier * delta)

    # Set our base intended price to the heuristic
    new_price = heuristic_price

    # -----------------------------------------------------------------
    # STEP 4: DEMAND MODEL ESTIMATION & NEXT PRICE OPTIMIZATION
    # -----------------------------------------------------------------
    # We require at least 5 periods of history to run a meaningful regression
    if t > 5 and prices_historical_in_current_season is not None and demand_historical_in_current_season is not None:
        try:
            # 1. Extract full history arrays
            p_own_hist = prices_historical_in_current_season[0, :]
            p_comp_hist = prices_historical_in_current_season[1, :]
            d_hist = demand_historical_in_current_season[:]
            
            # 2. Demand Model Estimation (OLS)
            N = len(d_hist)
            X = np.column_stack((np.ones(N), p_own_hist, p_comp_hist))
            y = d_hist
            
            # Solve for Beta coefficients using Least Squares
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            b0, b1, b2 = beta
            
            # 3. Next Price Optimization (Revenue Maximization)
            latest_comp_price = p_comp_hist[-1]
            
            # The model is only economically valid if our price coefficient (b1) is negative 
            if b1 < 0:
                # Calculate the mathematically optimal price to maximize immediate revenue
                optimal_price = -(b0 + b2 * latest_comp_price) / (2 * b1)
                optimal_price = np.clip(optimal_price, 20.0, 140.0) 
                
                # Blend 50% OLS optimization with 50% Heuristic baseline
                new_price = 0.5 * heuristic_price + 0.5 * optimal_price
                
        except Exception:
            # If matrix math fails, new_price remains heuristic_price
            pass
    # -----------------------------------------------------------------

    # 4. APPLY SAFETY NETS TO THE FINAL BLENDED PRICE
    time_ratio = t / MAX_PERIODS
    cap_ratio = remaining_capacity / TOTAL_CAPACITY
    if cap_ratio < (1.0 - time_ratio) * 0.75:  
        new_price *= 2.2
                    

    # Late season boost
    if t >= 85:
        late_season_boost = 1.0 + 0.25 * ((t - 85) / 15.0)
        new_price *= late_season_boost

    final_price = float(np.clip(new_price, 20.0, 140.0))
    information_dump['last_price'] = final_price
    
    return final_price, information_dump