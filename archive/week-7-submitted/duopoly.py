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

# Semaine 6 : prix optimisé sous contrainte de capacité (cours du 10/11, scipy.optimize)
USE_CAPACITY_OPT = True   # plan de prix des jours restants : max revenu attendu, ventes attendues <= sièges restants
CAPACITY_OPT_METHOD = 'lagrange'  # 'lagrange' (solution exacte par multiplicateur) ou 'slsqp' (scipy.optimize.minimize)
FOLLOW_BAND = True        # garder la correction par la courbe de remplissage en plus du plan optimisé

# Semaine 7 : approche d'apprentissage de Bertsimas & Perakis (cours du 17/11)
# Étapes 1-2 : la demande de LA compétition en cours est apprise au fil des saisons (information_dump gardé
# d'une saison à l'autre), en partant du modèle commun du .pkl (« a priori »).
USE_ONLINE_DEMAND = True 
ONLINE_MODE = 'level'     # 'level' : on apprend seulement le niveau de demande (a) ; 'full' : a, b et c
PRIOR_WEIGHT = 500.0      # poids du modèle commun, en nombre d'observations équivalentes (par bloc de 20 jours)
# Étapes 3-4 : prévoir la réaction du concurrent à notre prix et en tenir compte dans notre prix optimal.
# Modèle : prix_conc(t) - prix_conc(t-1) = alpha + k * (notre_prix(t-1) - prix_conc(t-1)), k entre 0 et 1
USE_REACTION = True 
REACTION_MIN_OBS = 30     # nombre minimum d'observations avant d'utiliser le modèle de réaction
REACTION_HORIZON = 20     # nombre de jours sur lesquels on compte l'effet de notre prix sur celui du concurrent

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


def get_ols_optimal_price(t: int, comp_price: float, default_price: float, learn: Optional[dict] = None,
                          gamma: float = 0.0, last_price: float = 0.0) -> float:
    # Modèle du bloc de 20 jours contenant t (corrigé par l'apprentissage et la réaction en semaine 7)
    a, b = effective_coefs(np.array([t]), comp_price, learn, gamma, last_price)
    a, b = float(a[0]), float(b[0])
    if b >= 0 or a <= 0:
        return default_price

    # Revenu R(p) = p * (a + b*p) : parabole concave, maximum en p* = -a / (2b)
    return float(np.clip(-a / (2.0 * b), MIN_PRICE, MAX_PRICE))

BLOCKS = (20, 40, 60, 80, 100)


def block_of(t: int) -> int:
    return next((k for k in BLOCKS if t <= k), BLOCKS[-1])


def get_ols_params(t: int, learn: Optional[dict] = None) -> Tuple[float, float, float]:
    # (a, b, c) du bloc de 20 jours contenant le jour t ; corrigés par l'apprentissage de la semaine 7
    k = block_of(t)
    params = OLS_MODELS[k]
    a, b, c = params['intercept'], params['price'], params['price_competitor']
    if learn is None or not USE_ONLINE_DEMAND:
        return a, b, c
    st = learn['blocks'][k]
    n = st['n']
    if n <= 0:
        return a, b, c
    prior = np.array([a, b, c])
    if ONLINE_MODE == 'full':
        # régression « ridge » vers le modèle commun : beta = (X'X + w*S)^-1 (X'y + w*S*prior), S = X'X/n
        S = st['xtx'] / n + np.diag([1e-6, 1e-6, 1e-6])
        w = PRIOR_WEIGHT
        try:
            beta = np.linalg.solve(st['xtx'] + w * S, st['xty'] + w * S @ prior)
        except np.linalg.LinAlgError:
            return a, b, c
        a, b, c = float(beta[0]), min(float(beta[1]), -0.005), max(float(beta[2]), 0.0)
        return a, b, c
    # 'level' : on décale seulement a de l'erreur moyenne du modèle commun dans cette compétition
    residual = st['xty'][0] - float(st['xtx'][0] @ prior)   # somme de (demande - prédiction du modèle commun)
    return a + residual / (n + PRIOR_WEIGHT), b, c


def learn_init() -> dict:
    return {'blocks': {k: {'n': 0, 'xtx': np.zeros((3, 3)), 'xty': np.zeros(3)} for k in BLOCKS},
            'react': np.zeros(5)}   # n, somme x, somme y, somme x², somme x*y


def learn_update(learn: dict, t_obs: int, price: float, comp_price: float, demand: float) -> None:
    # une observation de demande (jour t_obs, concurrent présent) : mise à jour de X'X et X'y du bloc
    x = np.array([1.0, price, comp_price])
    st = learn['blocks'][block_of(t_obs)]
    st['n'] += 1
    st['xtx'] += np.outer(x, x)
    st['xty'] += x * demand


def reaction_update(learn: dict, gap: float, change: float) -> None:
    # gap = notre prix - prix concurrent (veille) ; change = variation du prix concurrent le lendemain
    learn['react'] += np.array([1.0, gap, change, gap * gap, gap * change])


def reaction_params(learn: Optional[dict]) -> Optional[Tuple[float, float]]:
    # (alpha, k) estimés par moindres carrés, ou None si pas assez de données
    if learn is None or not USE_REACTION:
        return None
    n, sx, sy, sxx, sxy = learn['react']
    if n < REACTION_MIN_OBS:
        return None
    var = sxx - sx * sx / n
    if var <= 1e-9:
        return None
    k = float(np.clip((sxy - sx * sy / n) / var, 0.0, 1.0))
    alpha = (sy - k * sx) / n
    return float(alpha), k


def reaction_gamma(k: float, horizon: int) -> float:
    # part moyenne de notre changement de prix que le concurrent aura copiée sur les prochains jours
    # (jour 0 : il ne peut pas encore réagir ; jour j : 1 - (1-k)^j)
    h = max(1, horizon)
    j = np.arange(h)
    return float(np.mean(1.0 - (1.0 - k) ** j))


def effective_coefs(days: np.ndarray, comp_price: float, learn: Optional[dict], gamma: float,
                    last_price: float) -> Tuple[np.ndarray, np.ndarray]:
    """Demande attendue d_s = A_s + B_s * p (p = notre prix).
    Sans réaction : A = a + c*pc, B = b. Avec réaction (étape 4 du cours) : le concurrent suit une part gamma
    de notre changement de prix, pc(p) = pc + gamma*(p - dernier prix), donc A = a + c*(pc - gamma*dernier prix)
    et B = b + c*gamma (on garde B <= -0,002 pour que le revenu ait un maximum)."""
    abc = np.array([get_ols_params(int(s), learn) for s in days])
    a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
    if gamma > 0:
        g = np.minimum(gamma, np.where(c > 0, (-0.002 - b) / np.maximum(c, 1e-9), gamma))
        g = np.clip(g, 0.0, 1.0)
        return a + c * (comp_price - g * last_price), b + c * g
    return a + c * comp_price, b


def get_capacity_optimal_price(t: int, comp_price: float, remaining: int, learn: Optional[dict] = None,
                               gamma: float = 0.0, last_price: float = 0.0) -> Optional[float]:
    """Semaine 6 : plan de prix pour les jours t..100.
    max  sum_s p_s * d_s     avec d_s = a_s + b_s*p_s + c_s*pc  (pc = prix concurrent prédit, gardé constant)
    s.c. sum_s d_s <= sièges restants,  15 <= p_s <= 150,  d_s >= 0
    Semaine 7 : a, b, c appris dans la compétition, et réaction du concurrent (gamma).
    Renvoie le prix du jour t (None si le modèle n'est pas utilisable)."""
    days = np.arange(t, MAX_PERIODS + 1)
    a, b = effective_coefs(days, comp_price, learn, gamma, last_price)
    if np.any(b >= 0):
        return None
    lo, hi = MIN_PRICE, MAX_PRICE

    if CAPACITY_OPT_METHOD == 'slsqp':
        from scipy.optimize import minimize
        x0 = np.clip(-a / (2 * b), lo, hi)
        res = minimize(lambda x: -np.sum(x * (a + b * x)), x0, jac=lambda x: -(a + 2 * b * x),
                       method='SLSQP', bounds=[(lo, hi)] * len(days),
                       constraints=[{'type': 'ineq', 'fun': lambda x: remaining - np.sum(a + b * x),
                                     'jac': lambda x: -b[None, :]},
                                    {'type': 'ineq', 'fun': lambda x: a + b * x, 'jac': lambda x: np.diag(b)}],
                       options={'maxiter': 100})
        return float(np.clip(res.x[0], lo, hi)) if res.success else None

    # Méthode de Lagrange : chaque siège a une "valeur" lam >= 0 ; on maximise sum (p_s - lam) * d_s,
    # d'où p_s = p*_s + lam/2 (borné). lam = 0 si la contrainte ne gêne pas, sinon on cherche lam
    # (par dichotomie) pour que les ventes attendues = sièges restants.
    def plan(lam):
        return np.clip(-a / (2 * b) + lam / 2.0, lo, hi)
    def expected_sales(lam):
        return np.sum(np.maximum(0.0, a + b * plan(lam)))
    if expected_sales(0.0) <= remaining:
        return float(plan(0.0)[0])
    lam_lo, lam_hi = 0.0, 2 * (hi - lo)
    for _ in range(40):
        lam = 0.5 * (lam_lo + lam_hi)
        if expected_sales(lam) > remaining:
            lam_lo = lam
        else:
            lam_hi = lam
    return float(plan(lam_hi)[0])


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

    # Initialisation (l'apprentissage de la semaine 7 est gardé d'une saison à l'autre de la même compétition)
    if t == 1 or information_dump is None or not isinstance(information_dump, dict):
        learn = information_dump.get('learn') if isinstance(information_dump, dict) else None
        information_dump = {
            'sold_seats': 0,
            'last_price': 65.0,
            'random_curve': generate_random_curve() if RANDOM_TARGET_CURVE else None,
            'learn': learn if learn is not None else learn_init(),
            'comp_cap_history': [],
        }
    learn = information_dump.get('learn')
    caps = information_dump.setdefault('comp_cap_history', [])

    if t > 1 and demand_historical_in_current_season is not None:
        last_demand = demand_historical_in_current_season[-1]
        information_dump['sold_seats'] += int(last_demand)
        # Semaine 7 : on apprend de la veille (seulement si le concurrent avait des billets et nous aussi)
        if learn is not None and prices_historical_in_current_season is not None:
            try:
                ph = prices_historical_in_current_season
                if len(caps) >= 1 and caps[-1] and information_dump['sold_seats'] < TOTAL_CAPACITY:
                    learn_update(learn, t - 1, float(ph[0, -1]), float(ph[1, -1]), float(last_demand))
                if ph.shape[1] >= 2 and len(caps) >= 2 and caps[-1] and caps[-2]:
                    reaction_update(learn, float(ph[0, -2] - ph[1, -2]), float(ph[1, -1] - ph[1, -2]))
            except Exception:
                pass
    caps.append(bool(competitor_has_capacity_current_period_in_current_season))

    # Période 101 éventuelle (seulement pour transmettre la demande du jour 100) : pas de vrai prix à fixer
    if t > MAX_PERIODS:
        return float(information_dump.get('last_price', 65.0)), information_dump

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
        # Semaine 7, étapes 3-4 : le concurrent suit-il notre prix ? (pas en monopole)
        gamma, last_price = 0.0, float(prices_historical_in_current_season[0, -1])
        react = None if monopoly else reaction_params(learn)
        if react is not None:
            alpha, k = react
            last_comp = float(prices_historical_in_current_season[1, -1])
            comp_price = float(np.clip(last_comp + alpha + k * (last_price - last_comp), 0.0, MAX_PRICE))
            gamma = reaction_gamma(k, min(REACTION_HORIZON, MAX_PERIODS - t + 1))
        if comp_price > 0 and comp_price >= MIN_COMP_PRICE_OLS:
            ols_price = None
            if USE_CAPACITY_OPT:
                ols_price = get_capacity_optimal_price(t, comp_price, remaining_capacity, learn, gamma, last_price)
            if ols_price is None:
                ols_price = get_ols_optimal_price(t, comp_price, reference_price, learn, gamma, last_price)
            reference_price = (1.0 - OLS_WEIGHT) * reference_price + OLS_WEIGHT * ols_price

    # Ajustement seulement si on sort des marges de la courbe de remplissage
    if USE_CAPACITY_OPT and not FOLLOW_BAND:
        delta = 0.0
    elif sold_seats > upper:
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
