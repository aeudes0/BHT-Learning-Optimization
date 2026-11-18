# Journal de progression – DPC (Learning Optimization, BHT)

Traces pour les 4 présentations : ce qui a été fait, pourquoi, et les résultats obtenus.

| Présentation | Date | Thème demandé par le cours |
|---|---|---|
| #1 | 3 novembre | régression linéaire ; courbes de remplissage cibles vs courbes de ventes |
| #2 | 8 décembre | programmation dynamique ; prix optimisés |
| #3 | 19 janvier | machine learning (random forest, XGBoost…) |
| #4 | 2 février | meilleure performance |

## Versions de l'algorithme

| Version | Fichier archivé | Idée principale |
|---|---|---|
| W1/W2 initiale | `archive/week-2-submitted/duopoly_simple.py` | version simple (prix 66,4 / 33,2 visibles dans le CSV1) |
| W2 | `archive/week-2-submitted/duopolyW2.py` | suivre une courbe de remplissage cible (`target_sales_curve.pkl`), prix de référence = moyenne avec le prix concurrent |
| W3 (1re version) | `archive/week-2-submitted/duopolyW3.py` | modèle de demande OLS + optimisation scipy. **Bug** : l'optimiseur renvoyait presque toujours 150 € |
| W3 corrigée (soumise le 20/10) | `archive/week-3-submitted/duopoly_2026-10-20.py` | formule exacte `p* = -(a + c·pc) / (2b)`, poids réglés en simulation, plancher 15 € |
| W4 (soumise à partir du 31/10) | `duopoly.py` + `duopoly_params.pkl` | courbe de remplissage avec marges, OLS commun aux 16 compétitions (chargé depuis un .pkl), prévision ARIMA du prix concurrent, mode monopole basé sur le modèle |
| W6 (16/11, jamais soumise) | (incluse dans la W7) | W4 + prix optimisé sous contrainte de capacité (méthode de Lagrange), option `USE_CAPACITY_OPT` |
| W7 (18/11, à déposer) | `archive/week-7-submitted/` (`duopoly.py` + `duopoly_params.pkl`) | W6 + demande de la compétition apprise au fil des saisons + réaction du concurrent anticipée (Bertsimas & Perakis) |

## Résultats sur la plateforme DPC (`résultat_DPC/`)

| Run | Version | 1re place | Classement moyen | Revenu moyen / simulation | Ruptures | Concurrents |
|---|---|---|---|---|---|---|
| ≤ 16/10 (CSV1) | W1/W2 initiale | ? | ? | ≈ 174 100 $ | ? | SaffronDolphin, AttentiveOriole |
| 21/10 | W3 corrigée | 33 % | 1,67 | 149 106 $ | 23,2 % | RustlingNightingale, FlashyInchworm, DelicateMoose |
| 24/10 | W3 corrigée | 67 % | 1,33 | 58 762 $ | 1,0 % | ElectricButterfly, IndependentPorpoise, Winner_2023 |
| 31/10 | **W4** | **83 %** | 1,17 | 52 334 $ (classement général : **15e**) | 0 % | PompousRaven, PinkDuck, ElectricButterfly, ImpartialGoldfish, VersedJackdaw, UncoveredElephant |
| 2/11 | W4 | 56 % | 1,44 | 236 759 $ (classement général : **9e**) ; 2368 / saison, 57,8 sièges | 9,2 % | DaringVole, StoicCentipede, IndependentPorpoise, TopazCat, SageSloth, WhimsicalMarten, PinkDuck, IncredibleDogfish, ImposingCrocodile (18 compétitions) |
| 4/11 | W4 | 57 % | 1,43 | 136 107 $ (**11e**) ; 1361 / saison, 44,4 sièges | 7,4 % | 30 compétitions |
| 7/11 | W4 | 48 % | 1,52 | 198 828 $ (**9e**) ; 1988 / saison, 51,1 sièges | 13,3 % | 29 |
| 9/11 | W4 | 58 % | 1,42 | 217 630 $ (**8e**) ; 2176 / saison, 57,5 sièges | 10,5 % | 26 |
| 11/11 | W4 | 50 % | 1,50 | 183 386 $ (**13e**) ; 1834 / saison, 47,2 sièges | 10,3 % | 30 |
| 14/11 | W4 | 47 % | 1,53 | 165 921 $ (**12e**) ; 1659 / saison, 49,0 sièges | 3,5 % | 30 |
| 16/11 | W4 | 63 % | 1,37 | 206 284 $ (**11e**) ; 2063 / saison, 53,6 sièges | 10,6 % | 30 |

Depuis le 18/11, les CSV sont renommés `duopoly_competition_details<jour>_<mois>_<aa>.csv` ; `dpc_files.py` lit aussi l'année sur 2 chiffres.

Analyse du 2/11 (fichier nommé `details2_11_2026`, jour_mois au lieu de mois_jour : les scripts lisent maintenant les deux formats via `presentations/P1/dpc_files.py`) : 98,5 % des prix rejoués à l'identique avec la W4. Demande forte. Le nouveau mode monopole fonctionne : 2009 billets vendus quand le concurrent est en rupture, à 51 € en moyenne (contre 0 billet à 132 € avec la W3 le 21/10). Une compétition presque vide (4HKdWJ, WhimsicalMarten à 12,6 €) : 0,2 siège vendu par saison, prix bloqué au plancher de 15 €. 0 erreur, 19 ms par appel en moyenne (max 0,99 s).

Run d'Alois du 2/11 (W4, branche `alois` commit b3c028b, `results/duopoly_competition_details_1102.csv`) : 18 compétitions, 1751 / saison, 62,2 sièges, rupture de ses sièges dans 57 % des saisons, 0 erreur. Rejeu : 96 % des prix identiques avec sa W4 **sans** son `.pkl` (courbe linéaire de secours), contre 47 % avec le `.pkl`. Son code cherche `target_sales_curve.pkl` avec un chemin relatif, introuvable sur la plateforme : pas de plantage (le bug des clés texte ne s'est pas déclenché), mais sa vraie courbe cible n'a pas servi. Classement : à demander à Alois.

Comparaison équitable mise à jour avec le 2/11 (66 compétitions : 46 de Simon, 20 d'Alois). Sur les 50 jamais vues par le modèle de Simon W4 : Alois W4 2169, Alois W3 2167, Simon W4 2063, Simon W3 1950 (Alois W4 gagne 57 % des saisons, 7 % d'égalités).

Analyse du 31/10 : 99,5 % des prix rejoués à l'identique avec la W4. La demande était faible : seulement 27 sièges vendus sur 80 en moyenne, et la courbe de ventes reste sous la marge basse dans 100 % des saisons. Le prix descend donc souvent au plancher de 15 € (jusqu'à 93 % des périodes contre ElectricButterfly). Leçon : la courbe cible doit s'adapter au niveau de demande.

Algorithme d'Alois (branche `alois`) : en W4, modèle ARX(1) estimé pendant la saison (régression en ligne), mélangé à 40 % avec la courbe cible. Résultats DPC (d'après son `Project_log.md`) : W2 16e et 67e sur 75 ; W3 46e et 43e sur 75. Simulation sur 28 compétitions : Simon W4 1366, Alois W4 1228, Simon W3 1163 ; Simon W4 fait mieux dans 52 % des saisons ; Alois est légèrement devant sur le run du 31/10.

**Comparaison équitable (1/11, `presentations/P1/team_analysis.py`)** : 48 compétitions (28 de Simon + 20 d'Alois, branche `alois`, dossier `results/`). Revenu simulé par saison :
- sur tout : Simon W4 1825, Alois W3 1806, Alois W4 1760, Simon W3 1616 ;
- sur les 32 compétitions jamais vues par le modèle de Simon W4 : Alois W4 **1774**, Alois W3 1733, Simon W4 1697, Simon W3 1585 (Alois W4 gagne 65 % des saisons).

Chaque algo gagne sur les données de son auteur ; sur des données nouvelles, l'apprentissage en ligne est un peu devant. Dans les deux algos, la phase de monopole est gâchée (0 à 3 billets vendus).
⚠️ Le `target_sales_curve.pkl` de la W4 d'Alois a des clés texte (format du notebook W4 du prof) : avec lui, `duopoly.py` lève une `TypeError` à chaque mise à jour. La simulation utilise ses clés converties en nombres.

**Présentation #1 (3/11)** : 18 diapos en anglais, avec notes d'orateur et répartition Simon/Alois. Version finale (données du 2/11 incluses) : https://claude.ai/artifact/JzApbTBuTXZ6M8VqgNk59f — graphiques générés par `presentations/P1/make_figures.py` dans `presentations/P1/figures/`.

Le revenu dépend surtout du concurrent et de la demande tirée au hasard : seul le classement permet de comparer les runs.

## Constats après les runs du 21/10 et du 24/10 (analyse du 27/10)
- Le modèle OLS hardcodé vient d'une seule compétition (`3HmCSH`, prix bas). Il recommande 16-25 €, alors que les modèles ré-estimés sur chaque nouvelle compétition donnent 30-50 € (run du 21/10). Son erreur de prédiction est 2 à 3 fois plus grande que celle d'un modèle ré-estimé.
- Rejouées dans les 12 nouvelles compétitions (simulation), les versions donnent : W2 = 2102, W3 soumise = 1843, W3 avec le vrai modèle de chaque compétition = 2417 (revenu moyen par saison, run du 21/10). L'idée OLS est bonne, mais il faut un modèle plus général que celui d'une seule compétition.
- Mode monopole : 10 334 périodes avec le concurrent en rupture au 21/10, **0 billet vendu** à 132 € en moyenne, alors qu'il nous restait 13,6 sièges en moyenne au jour 79.
- 24/10 : face à des concurrents presque gratuits (Winner_2023 à 0,67 € en moyenne), on reste au plancher de 15 € et on vend 6 à 17 sièges par saison, mais on gagne quand même 2 compétitions sur 3.
- Test ARIMA(1,1,1) pour prédire le prix concurrent du lendemain : erreur moyenne 2,50 € contre 2,47 € pour « le prix d'hier », donc pas mieux sur une seule saison. Temps de calcul : 26 ms en moyenne.

## Semaine 4 (27/10) – `W4-Notebooks/W4_capacity_OLS_ARIMA.ipynb`
- **Courbes randomisées** : cible = moyenne des ventes cumulées des 20 % meilleures saisons (16 compétitions) ; marges = quantiles 25 % et 75 % (au jour 50 : 31 / 41 / 52 sièges). Le prix n'est corrigé qu'en dehors des marges.
- **OLS commun** (16 compétitions, saisons 1-50, hors périodes où le concurrent est en rupture) : erreur sur les saisons 51-100 de 0,60-0,73 contre 0,67-1,51 pour l'ancien modèle ; prix optimal de 33-70 € au lieu de 16-25 €. R² faible (0,01-0,07), car la demande par jour est très aléatoire (Poisson).
- **ARIMA(1,1,1)** : ne prédit pas mieux que « le prix d'hier » (2,50 € d'erreur contre 2,47 €), sauf pour RustlingNightingale. Effet sur le revenu simulé : nul (-0,2 € par saison).
- **Simulation** sur les 16 compétitions (revenu moyen par saison) :

| Version | 21/10 | 24/10 | avant 16/10 | Moyenne |
|---|---|---|---|---|
| W3 soumise | 2628 | 599 | 1967 | 1702 |
| **W4** (marges) | **3466** | 619 | 2258 | **2096** |
| W4 sans marges (même courbe cible) | 3404 | 598 | 2265 | 2067 |
| W4 avec l'ancien mode monopole | 3170 | 680 | 2179 | 1989 |
| W4 courbe aléatoire (méthode du cours), sans marges | 3436 | 600 | 2242 | 2074 |
| W4 courbe aléatoire + marges | 3464 | 618 | 2199 | 2080 |

  Le gain vient surtout de l'OLS commun et du mode monopole. Les marges apportent environ +1,4 %. (Une première comparaison avait indiqué -1 %, mais la version « sans marges » utilisait par erreur l'ancienne courbe cible de la semaine 3.) Le plancher de 15 € et `OLS_WEIGHT = 1` restent les meilleurs réglages.
- **Méthode du cours pour les courbes randomisées** (`W4_FULL_dynamic_utilization_curves`) : pas de 5 jours suivant N(μ = 0,050, σ = 0,006), cumulés. Les courbes tirées ne s'écartent que de ±1,4 siège au jour 50 (les pas aléatoires se compensent), d'où un effet quasi nul sur le revenu et sur la variété des prix (écart-type du prix moyen : 4,8 contre 4,7).
- Test « plateforme » : `duopoly.py` + `duopoly_params.pkl` dans un dossier vide ; 11,5 ms par appel en moyenne, 155 ms au maximum ; sans le .pkl, retour aux valeurs de la semaine 3.

## Semaine 6 (cours du 10/11, travail du 16/11) – `W6-Notebooks/W6_constrained_optimization.ipynb`
Thème : optimisation sous contraintes (scipy.optimize, Pyomo), appliquée à la régression et au prix. Pyomo + HiGHS installés dans le venv (`pip install pyomo highspy`) ; ipopt pas nécessaire (HiGHS résout les problèmes quadratiques si l'objectif est écrit en polynôme).
- **Régression avec Pyomo** (`w6_regression.py`) : OLS Pyomo = statsmodels à l'identique. Les contraintes `b ≤ -0,005, c ≥ 0` ne changent rien (déjà respectées sur le modèle commun). **LAD inutilisable** : 60 % des jours à 0 vente, donc LAD trouve a = b = c = 0 (prédit la médiane, 0).
- **Prix sous contrainte de capacité** (nouvelle fonction `get_capacity_optimal_price` dans `duopoly.py`) : maximiser le revenu attendu des jours restants avec ventes attendues ≤ sièges restants, 15-150 €. Solution de Lagrange : `p = p* + λ/2` (λ = valeur d'un siège, 0 si les sièges suffisent). SLSQP (scipy), Pyomo et Lagrange donnent le même prix ; Lagrange 0,3-0,5 ms contre 5,7 ms pour SLSQP. Exemple jour 30, concurrent à 50 € : 69,9 € avec 80 ou 40 sièges, 98,9 € avec 20, 111,4 € avec 10.
- **Simulation** (`w6_simulation.py`, 84 compétitions dont le 2/11 d'Alois, saisons 51-99) : W4 2045 / saison ; **W6 optimisé + marges 2055** (+0,5 %, gagne 11,6 % des saisons, en perd 2,5 %) ; W6 optimisé sans marges 1888 (excellent en forte demande, ex. 21/10 d'Alois 3909 contre 3755, mais catastrophique en faible demande, 24/10 d'Alois 403 contre 871).
- Leçon : la contrainte de capacité n'aide que si le modèle connaît le niveau de demande de la compétition en cours ; le modèle commun ne le connaît pas, les marges compensent. Suite logique : apprentissage pendant la saison (cours du 17/11).
- `USE_CAPACITY_OPT = True` (avec `FOLLOW_BAND = True`) activé localement, **pas encore soumis**. Test « plateforme » (dossier vide, avec ARIMA) : 0 erreur, 46 ms par appel en moyenne, 506 ms au maximum.

## Semaine 7 (cours du 17/11, travail du 18/11) – `W7-Notebooks/W7_learning_approach.ipynb`
Thème : approche d'apprentissage de Bertsimas & Perakis (1. apprendre notre demande, 2. moyenne des β récents, 3. modèle du concurrent, 4. prix optimal qui anticipe le concurrent). Code du prof en Pyomo + glpk + ipopt : pas utilisable sur DPC, tout est refait en numpy.
- **Réaction des concurrents** (`w7_competitor_reaction.py`, 217 compétitions des runs de Simon) : 102 indépendants, 41 suiveurs faibles, 55 suiveurs forts, **19 copieurs exacts** (prix = notre prix d'hier). Contrôle placebo (notre prix de demain) bien plus faible : la réaction est réelle.
- **Étapes 1-2** (`w7_demand_learning.py`, 18 compétitions jamais vues) : MSE de la demande du jour : a, b, c appris 0,604 ; niveau appris 0,614 ; modèle commun 0,660 ; LAD lisse du cours 0,762 (meilleure MAE mais vise la médiane, comme en W6).
- **Code** : `information_dump['learn']` gardé d'une saison à l'autre (X'X et X'y par bloc de 20 jours ; sommes pour la réaction) ; `ONLINE_MODE = 'level'` (a corrigé avec un poids a priori de 500 observations) ; réaction `Δpc = α + k·(p − pc)` (k entre 0 et 1, après 30 observations), prix concurrent prévu par ce modèle au lieu d'ARIMA, pente effective `b + c·γ` (γ = part de notre changement suivie sur 20 jours) dans le prix optimal et dans le plan sous contrainte de capacité. Période 101 gérée.
- **Simulation** (`w7_simulation.py`, 259 compétitions, 30 saisons à la suite), concurrent réactif / rejoué : W4 2165 / 2071 ; W6 2189 / 2091 ; niveau appris 2292 / 2156 ; **niveau + réaction 2715 (+25 %) / 2011 (−3 %)** ; sans marges 3453 / 1583 (trop risqué).
- Choix pour la W7 déposée : niveau appris + réaction + marges. Test « plateforme » : 10 saisons à la suite avec un concurrent copieur : k appris = 1,00, 0 erreur, 2,2 ms par appel en moyenne (74 ms max), information_dump de 1,3 ko.
- À surveiller sur DPC : classement et part de 1res places ; si la réaction fait perdre, mettre `USE_REACTION = False` (on garde le gain plus sûr du niveau appris).

## Décisions et pistes
- Ligne de conduite : suivre le cours semaine par semaine ; garder les améliorations hors cours pour plus tard.
- Couvert plus tard par le cours : exploiter la rupture du concurrent (15/12), regrouper les compétitions par type de demande (5/01), approche d'apprentissage de Bertsimas & Perakis (17/11).
