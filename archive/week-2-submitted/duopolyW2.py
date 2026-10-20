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
    
    if t == 1 or information_dump is None or not isinstance(information_dump, dict):
        target_curve = load_target_curve()
        information_dump = {
            'sold_seats': 0,
            'last_price': 65.0,  # Higher initial baseline price
            'target_curve': target_curve
        }
    
    target_curve = information_dump.get('target_curve', load_target_curve())
    
    if t > 1 and demand_historical_in_current_season is not None:
        last_demand = demand_historical_in_current_season[-1]
        information_dump['sold_seats'] += int(last_demand)
    
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
        if comp_price >= 25.0:
            reference_price = 0.5 * reference_price + 0.5 * comp_price

    # Calculate percentage deviation from target curve
    delta = (sold_seats - target_sales) / max(1.0, target_sales)
    
    # If delta > 0 (selling too fast), increase price more aggressively (multiplier 2)
    # If delta < 0 (selling too slow), lower price to pick up sales (multiplier 0.5)
    pricing_multiplier = 2 if delta > 0 else 0.5
    new_price = reference_price * (1.0 + pricing_multiplier * delta)
    
    # Extra safety net: if inventory is running out way too early compared to remaining time
    time_ratio = t / MAX_PERIODS
    cap_ratio = remaining_capacity / TOTAL_CAPACITY
    if cap_ratio < (1.0 - time_ratio) * 0.7:  # We are burning inventory too fast
        new_price *= 3                     # Force a strong price hike

    # ---- PONDERATION POUR LA FORTE DEMANDE DE FIN DE PÉRIODE ----
    # À partir du jour 85, la demande explose naturellement vers le départ (jour 100).
    # On applique un coefficient multiplicateur progressif pour valoriser les derniers sièges.
    if t >= 85:
        # Le coefficient augmente doucement de 1.0 (au jour 85) jusqu'à ~1.3 (au jour 100)
        late_season_boost = 1.0 + 0.2 * ((t - 85) / 15.0)
        new_price *= late_season_boost

    final_price = float(np.clip(new_price, 30.0, 150.0))
    information_dump['last_price'] = final_price
    
    return final_price, information_dump