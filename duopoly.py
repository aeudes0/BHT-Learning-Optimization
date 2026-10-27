from typing import Any, Optional, Tuple, Union
import os
import pickle
import warnings
import numpy as np

try:
    # importé au chargement pour que le premier appel de p() ne soit pas ralenti (~1,5 s)
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

TOTAL_CAPACITY = 80
MAX_PERIODS = 100

# Hyperparamètres (modifiables pour les tests en simulation)
OLS_WEIGHT = 1.0          # poids du prix OLS dans le lissage (1 - OLS_WEIGHT pour le dernier prix)
MULTIPLIER_UP = 1.0       # réaction quand on vend plus vite que la marge haute
MULTIPLIER_DOWN = 1.0     # réaction quand on vend moins vite que la marge basse
BOOST_START = 95          # jour de début de la majoration de fin de saison
BOOST_AMPLITUDE = 0.0     # majoration maximale atteinte au jour 100 (0 = désactivée)
MIN_PRICE = 15.0          # prix plancher (aucune règle de la plateforme, choix stratégique)
MAX_PRICE = 150.0         # prix plafond (les clients paient au plus 150)
MIN_COMP_PRICE_OLS = 0.0  # en dessous de ce prix concurrent, on n'utilise pas le prix OLS

# Semaine 4
USE_CAPACITY_BAND = True  # courbe de remplissage avec marges (sinon courbe cible seule)
RANDOM_TARGET_CURVE = False  # méthode du cours : nouvelle courbe cible tirée au hasard à chaque saison
USE_ARIMA = True          # prédire le prix concurrent avec ARIMA (sinon dernier prix observé)
ARIMA_ORDER = (1, 1, 1)
ARIMA_MIN_HISTORY = 10    # nombre minimum de prix observés avant d'utiliser ARIMA
MONOPOLY_USE_MODEL = True # concurrent en rupture : prix OLS avec un concurrent "à 150" (sinon ancienne règle)

# Fichier de paramètres produit par W4-Notebooks/W4_capacity_OLS_ARIMA.ipynb
PARAMS_FILE = 'duopoly_params.pkl'

# Valeurs de secours si le fichier .pkl est absent (paramètres de la semaine 3)
TARGET_CURVE = {
    1: 1, 6: 4, 11: 6, 16: 10, 21: 13, 26: 17, 31: 21, 36: 25, 41: 29, 46: 33,
    51: 36, 56: 40, 61: 43, 66: 47, 71: 51, 76: 56, 81: 60, 86: 64, 91: 67, 96: 68
}
OLS_MODELS = {
    20: {'price': -0.0299, 'price_competitor': 0.0115, 'intercept': 0.9819},
    40: {'price': -0.0604, 'price_competitor': 0.0467, 'intercept': 0.8657},
    60: {'price': -0.1449, 'price_competitor': 0.1353, 'intercept': 0.9659},
    80: {'price': -0.0971, 'price_competitor': 0.0907, 'intercept': 1.0316},
    100: {'price': -0.0794, 'price_competitor': 0.0588, 'intercept': 1.1548}
}
CAPACITY_BAND = None      # {t: {'lower', 'target', 'upper'}} chargé depuis le .pkl
# Loi normale des augmentations de la courbe cible tous les 5 jours (en part de la capacité)
_steps = np.diff([TARGET_CURVE[d] for d in sorted(TARGET_CURVE)]) / TOTAL_CAPACITY
CURVE_STEPS = {'mu': float(np.mean(_steps)), 'std': float(np.std(_steps))}


def load_params():
    # Sur la plateforme, les fichiers déposés sont dans /root/partcipant_folder/ (orthographe de la doc DPC)
    folders = ['/root/partcipant_folder/', '/root/participant_folder/',
               os.path.dirname(os.path.abspath(__file__)), '.']
    for folder in folders:
        try:
            with open(os.path.join(folder, PARAMS_FILE), 'rb') as f:
                return pickle.load(f)
        except Exception:
            continue
    return None


_params = load_params()
if _params is not None:
    OLS_MODELS = _params.get('ols_models', OLS_MODELS)
    CAPACITY_BAND = _params.get('capacity_band', CAPACITY_BAND)
    CURVE_STEPS = _params.get('curve_steps', CURVE_STEPS)


def generate_random_curve() -> np.ndarray:
    # Courbe cible aléatoire (notebook W4 du cours) : 20 pas de 5 jours tirés dans N(mu, std),
    # cumulés, puis interpolés jour par jour ; renvoie les sièges cumulés pour les jours 1 à 100
    steps = np.clip(np.random.normal(CURVE_STEPS['mu'], CURVE_STEPS['std'], size=20), 0.0, None)
    days = np.arange(1, 102, 5)                      # 1, 6, ..., 101
    curve = np.concatenate([[0.0], np.cumsum(steps)])
    curve = np.minimum(curve, 1.0) * TOTAL_CAPACITY
    return np.interp(np.arange(1, MAX_PERIODS + 1), days, curve)


def get_ols_optimal_price(t: int, comp_price: float, default_price: float) -> float:
    # Sélection du modèle correspondant à la période actuelle
    params = OLS_MODELS[100]
    for p_max in [20, 40, 60, 80, 100]:
        if t <= p_max:
            params = OLS_MODELS[p_max]
            break

    b = params['price']
    a = params['intercept'] + params['price_competitor'] * comp_price
    if b >= 0 or a <= 0:
        return default_price

    # Revenu R(p) = p * (a + b*p) : parabole concave, maximum en p* = -a / (2b)
    return float(np.clip(-a / (2.0 * b), MIN_PRICE, MAX_PRICE))

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

def get_capacity_band(t: int, random_curve: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    # (marge basse, cible, marge haute) en sièges vendus cumulés
    if CAPACITY_BAND is not None and t in CAPACITY_BAND:
        band = CAPACITY_BAND[t]
        lower, target, upper = float(band['lower']), float(band['target']), float(band['upper'])
    else:
        lower = target = upper = get_target_sales(t)
    if random_curve is not None:
        # la courbe aléatoire de la saison remplace la cible, les marges gardent leur écart à la cible
        shift = float(random_curve[t - 1]) - target
        lower, target, upper = lower + shift, target + shift, upper + shift
    if not USE_CAPACITY_BAND:
        return target, target, target
    return lower, target, upper

def predict_competitor_price(comp_prices: np.ndarray) -> float:
    # Prévision ARIMA du prix concurrent de la période suivante (dernier prix si impossible)
    last_price = float(comp_prices[-1])
    if (not USE_ARIMA or ARIMA is None or len(comp_prices) < ARIMA_MIN_HISTORY
            or np.all(comp_prices == comp_prices[0])):
        return last_price
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            forecast = ARIMA(np.asarray(comp_prices, dtype=float), order=ARIMA_ORDER).fit().forecast(1)[0]
        if not np.isfinite(forecast):
            return last_price
        return float(np.clip(forecast, 0.0, MAX_PRICE))
    except Exception:
        return last_price

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
            'random_curve': generate_random_curve() if RANDOM_TARGET_CURVE else None,
        }

    if t > 1 and demand_historical_in_current_season is not None:
        last_demand = demand_historical_in_current_season[-1]
        information_dump['sold_seats'] += int(last_demand)

    sold_seats = information_dump['sold_seats']
    remaining_capacity = max(0, TOTAL_CAPACITY - sold_seats)

    # Protection contre la rupture de stock
    if remaining_capacity <= 0:
        return 150.0, information_dump

    monopoly = not competitor_has_capacity_current_period_in_current_season

    # Mode monopole (ancienne règle) si le concurrent n'a plus de billets
    if monopoly and not MONOPOLY_USE_MODEL:
        monopoly_price = 80.0 + 60.0 * (t / MAX_PERIODS)
        price = float(np.clip(monopoly_price, 40.0, 150.0))
        information_dump['last_price'] = price
        return price, information_dump

    # Prix de base pour le premier jour
    if t == 1:
        base_price = 65.0
        information_dump['last_price'] = base_price
        return base_price, information_dump

    # Fréquence de mise à jour des prix (tous les jours en monopole)
    should_update = monopoly
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

    lower, target_sales, upper = get_capacity_band(t, information_dump.get('random_curve'))
    reference_price = information_dump.get('last_price', 65.0)

    # Prix OLS optimal avec le prix concurrent prédit, lissé avec le prix de référence
    if prices_historical_in_current_season is not None:
        if monopoly:
            # concurrent en rupture : comme un concurrent plus cher que ce que les clients acceptent
            comp_price = MAX_PRICE
        else:
            comp_price = predict_competitor_price(prices_historical_in_current_season[1])
        if comp_price > 0 and comp_price >= MIN_COMP_PRICE_OLS:
            ols_price = get_ols_optimal_price(t, comp_price, reference_price)
            reference_price = (1.0 - OLS_WEIGHT) * reference_price + OLS_WEIGHT * ols_price

    # Ajustement seulement si on sort des marges de la courbe de remplissage
    if sold_seats > upper:
        delta = (sold_seats - upper) / max(1.0, upper)
    elif sold_seats < lower:
        delta = (sold_seats - lower) / max(1.0, lower)
    else:
        delta = 0.0
    pricing_multiplier = MULTIPLIER_UP if delta > 0 else MULTIPLIER_DOWN
    new_price = reference_price * (1.0 + pricing_multiplier * delta)

    # Sécurité supplémentaire si le stock s'écoule trop vite
    time_ratio = t / MAX_PERIODS
    cap_ratio = remaining_capacity / TOTAL_CAPACITY
    if cap_ratio < (1.0 - time_ratio) * 0.7:
        new_price *= 3

    # Majoration de fin de saison
    if t >= BOOST_START:
        late_season_boost = 1.0 + BOOST_AMPLITUDE * ((t - BOOST_START) / (MAX_PERIODS - BOOST_START))
        new_price *= late_season_boost

    final_price = float(np.clip(new_price, MIN_PRICE, MAX_PRICE))
    information_dump['last_price'] = final_price

    return final_price, information_dump
