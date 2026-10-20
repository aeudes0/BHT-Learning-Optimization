from typing import Any, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt

TOTAL_CAPACITY = 80
MAX_PERIODS = 100

# Courbe de ventes cible (sièges vendus cumulés par période)
TARGET_CURVE = {
    1: 1, 6: 4, 11: 6, 16: 10, 21: 13, 26: 17, 31: 21, 36: 25, 41: 29, 46: 33, 
    51: 36, 56: 40, 61: 43, 66: 47, 71: 51, 76: 56, 81: 60, 86: 64, 91: 67, 96: 68
}

# Paramètres du modèle de demande OLS par tranches de 20 jours
OLS_MODELS = {
    20: {'price': -0.0299, 'price_competitor': 0.0115, 'intercept': 0.9819},
    40: {'price': -0.0604, 'price_competitor': 0.0467, 'intercept': 0.8657},
    60: {'price': -0.1449, 'price_competitor': 0.1353, 'intercept': 0.9659},
    80: {'price': -0.0971, 'price_competitor': 0.0907, 'intercept': 1.0316},
    100: {'price': -0.0794, 'price_competitor': 0.0588, 'intercept': 1.1548}
}

def get_ols_optimal_price(t: int, comp_price: float, default_price: float) -> float:
    # Sélection du modèle correspondant à la période actuelle
    for p_max in [20, 40, 60, 80, 100]:
        if t <= p_max:
            params = OLS_MODELS[p_max]
            break
            
    if params['price'] >= 0:
        return default_price

    # Fonction de revenu négatif à minimiser pour maximiser le profit
    def neg_revenue(price):
        demand = params['intercept'] + params['price'] * price + params['price_competitor'] * comp_price
        demand = max(0.0, demand)
        return -(price * demand)

    res = opt.minimize_scalar(neg_revenue, bounds=(30.0, 150.0), method='bounded')
    return float(res.x)

def get_target_sales(t: int) -> float:
    days = sorted(TARGET_CURVE.keys())
    if t in TARGET_CURVE:
        return float(TARGET_CURVE[t])
    for i in range(len(days) - 1):
        d1, d2 = days[i], days[i + 1]
        if d1 <= t <= d2:
            s1 = TARGET_CURVE[d1]
            s2 = TARGET_CURVE[d2]
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
    
    # Initialisation
    if t == 1 or information_dump is None or not isinstance(information_dump, dict):
        information_dump = {
            'sold_seats': 0,
            'last_price': 65.0,
        }
    
    if t > 1 and demand_historical_in_current_season is not None:
        last_demand = demand_historical_in_current_season[-1]
        information_dump['sold_seats'] += int(last_demand)
    
    sold_seats = information_dump['sold_seats']
    remaining_capacity = max(0, TOTAL_CAPACITY - sold_seats)
    
    # Protection contre la rupture de stock
    if remaining_capacity <= 0:
        return 150.0, information_dump

    # Mode monopole si le concurrent n'a plus de billets
    if not competitor_has_capacity_current_period_in_current_season:
        monopoly_price = 80.0 + 60.0 * (t / MAX_PERIODS)
        price = float(np.clip(monopoly_price, 40.0, 150.0))
        information_dump['last_price'] = price
        return price, information_dump

    # Prix de base pour le premier jour
    if t == 1:
        base_price = 65.0
        information_dump['last_price'] = base_price
        return base_price, information_dump

    # Fréquence de mise à jour des prix
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

    target_sales = get_target_sales(t)
    reference_price = information_dump.get('last_price', 65.0)
    
    # Calcul du prix optimal OLS et lissage avec le prix de référence
    if prices_historical_in_current_season is not None:
        comp_price = prices_historical_in_current_season[1, -1]
        if comp_price >= 25.0:
            ols_price = get_ols_optimal_price(t, comp_price, reference_price)
            reference_price = 0.5 * reference_price + 0.5 * ols_price

    # Ajustement selon l'écart avec la courbe cible
    delta = (sold_seats - target_sales) / max(1.0, target_sales)
    pricing_multiplier = 2 if delta > 0 else 0.5
    new_price = reference_price * (1.0 + pricing_multiplier * delta)
    
    # Sécurité supplémentaire si le stock s'écoule trop vite
    time_ratio = t / MAX_PERIODS
    cap_ratio = remaining_capacity / TOTAL_CAPACITY
    if cap_ratio < (1.0 - time_ratio) * 0.7:
        new_price *= 3

    # Majoration de fin de saison
    if t >= 85:
        late_season_boost = 1.0 + 0.2 * ((t - 85) / 15.0)
        new_price *= late_season_boost

    final_price = float(np.clip(new_price, 30.0, 150.0))
    information_dump['last_price'] = final_price
    
    return final_price, information_dump