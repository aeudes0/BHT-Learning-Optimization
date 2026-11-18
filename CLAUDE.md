# Contexte du Projet - Competition Dynamic Pricing (BHT)

## 0. Façon de travailler (demandes de Simon)
- **Répondre en français.** Les présentations sont en anglais (anglais simple : Simon et Alois ne sont pas bilingues).
- **Aucun commit ni push** : modifications et tests en local uniquement, puis donner les commandes git à lancer dans Git Bash.
- **Suivre le cours semaine par semaine** (`doc/coursN`). Les améliorations hors cours sont proposées, mais gardées pour plus tard ; c'est Simon qui décide. Ne pas utiliser une notion avant le cours qui l'introduit.
- Cycle hebdomadaire : analyser les résultats DPC et les notebooks → comprendre ce qui a échoué → lire le cours → proposer → Simon décide.
- Rester objectif et équilibré entre Simon et Alois (le binôme) dans les analyses et les diapos.

## 1. Objectif du Projet
Bot de tarification pour la compétition de pricing en duopole (`duopoly.py`), évalué par **4 présentations de groupe** (note = moyenne des 3 meilleures), à deux avec **Alois** (branche `alois`).

- Présentations : #1 le 3/11 (régression linéaire, courbes cibles vs courbes de ventes), #2 le 8/12 (programmation dynamique, prix optimisés), #3 le 19/01 (ML : RF, XGBoost), #4 le 2/02 (meilleure performance). Planning complet : `doc/sujets des différents cours.txt`.
- Chaque présentation : état actuel, objectifs, détails d'implémentation, ce qui a marché ou non, évaluation des performances (~20 min).
- Suivi : **`JOURNAL_PROGRESSION.md`** (versions, résultats DPC, constats, décisions) ; chaque version soumise est copiée dans `archive/week-N-submitted/`.

## 2. Règles de la compétition (doc/cours1, doc/cours4)
- **Plateforme** : www.dynamic-pricing-competition.com. On dépose `duopoly.py` (fonction `p(...)` qui renvoie `(prix, information_dump)`). Depuis le cours 4, on peut aussi déposer des fichiers annexes (`.pkl`), lus depuis `/root/partcipant_folder/` (orthographe de la doc DPC). Runs lundi, mercredi et samedi à 0h30 UTC.
- **Structure** : chaque run affronte plusieurs concurrents (de plus en plus au fil des semaines : 2 → 9), **2 simulations par concurrent** ; 1 simulation (= 1 `competition_id` du CSV) = 100 saisons de 100 jours, 80 sièges. « placement1 » sur le site = % de simulations gagnées (duopole : 1er ou 2e).
- **Demande** : tirée au hasard par simulation, arrivées de Poisson, clients prêts à payer au plus 150.
- **Contraintes** : Python 3.12.9 + Anaconda, 1 CPU / 512 Mo, < 0,2 s en moyenne par appel (max 5 s), ≥ 99 % d'appels sans erreur, pas d'écriture de fichiers.
- **Classement** : la victoire face au concurrent compte, pas le revenu seul. Résumés du site : `résultat_DPC/resultat.txt`.

## 3. Algorithme de Simon (`duopoly.py` + `duopoly_params.pkl`, version W4)
- **Prix de référence** : OLS `demande = a + b·prix + c·prix_concurrent` (5 blocs de 20 jours), `p* = -(a + c·prix_concurrent) / (2b)`, borné entre 15 et 150. Prix concurrent prédit par **ARIMA(1,1,1)** (dernier prix si < 10 observations).
- **Courbe de remplissage avec marges** : cible = moyenne des ventes cumulées des 20 % meilleures saisons ; marges = quantiles 25 % / 75 %. Prix corrigé seulement hors des marges.
- **Mode monopole** (concurrent en rupture) : prix OLS avec un concurrent « à 150 », mis à jour chaque jour.
- **Options** : `USE_CAPACITY_BAND = True`, `RANDOM_TARGET_CURVE = False`, `USE_ARIMA = True`, `MONOPOLY_USE_MODEL = True` ; `OLS_WEIGHT = 1.0`, `MIN_PRICE = 15.0`, `MAX_PRICE = 150.0`.
- **`duopoly_params.pkl`** (produit par `W4-Notebooks/W4_capacity_OLS_ARIMA.ipynb`, entraîné sur les 16 premières simulations : avant 16/10, 21/10, 24/10) : `ols_models`, `capacity_band`, `curve_steps`. Coefficients jours 1-20 : a = 0,635, b = -0,0115, c = 0,0059.
- **W6 (16/11, pas encore soumise)** : `USE_CAPACITY_OPT = True` : le prix OLS devient le prix du jour d'un plan optimisé sur les jours restants (revenu attendu max, ventes attendues ≤ sièges restants), résolu par Lagrange (`p = p* + λ/2`, dichotomie sur λ, 0,5 ms) ; `CAPACITY_OPT_METHOD = 'slsqp'` possible (plus lent). `FOLLOW_BAND = True` : les marges restent indispensables (sans elles, désastre en faible demande). Gain simulé +0,5 %. Détails : `W6-Notebooks/`.
- **W7 (18/11, copie dans `archive/week-7-submitted/`)** : en plus de la W6, `USE_ONLINE_DEMAND = True` (`ONLINE_MODE = 'level'`, `PRIOR_WEIGHT = 500` : niveau de demande appris dans la compétition, `information_dump['learn']` gardé entre les saisons) et `USE_REACTION = True` (modèle `Δpc = α + k·(p − pc)` appris en ligne ; prix concurrent prévu par ce modèle ; pente effective `b + c·γ`). Simulation : +25 % si le concurrent réagit, −3 % s'il ne réagit pas. Détails : `W7-Notebooks/` et le journal.
- **À déposer sur DPC : `duopoly.py` ET `duopoly_params.pkl`.**

## 4. Algorithme d'Alois (branche `alois`)
- W3 : OLS ré-estimé pendant la saison, 50 % p* + 50 % prix de la courbe.
- W4 : ARX(1) `demande = b0 + b1·prix + b2·prix_conc + φ·demande_hier`, estimé avec `np.linalg.lstsq` sur la saison en cours à partir du jour 7 ; p* utilisé seulement si b1 < -0,01 ; prix = 40 % p* + 60 % courbe.
- **Bugs à lui signaler** : (1) son `target_sales_curve.pkl` W4 a des clés texte → `TypeError` si le fichier est chargé ; (2) il le cherche avec un chemin relatif, introuvable sur DPC → le 2/11 il a tourné avec la courbe linéaire de secours (96 % des prix rejoués). Correctif : chercher dans `/root/partcipant_folder/` et convertir les clés avec `int(k)`.
- Hypothèse non vérifiée : le seuil b1 < -0,01 est souvent non atteint en début de saison, donc la partie « online » servirait moins souvent que prévu.
- Copies locales de sa branche (ignorées par git) : `BHT-Learning-Optimization_Alois/alois_b3c028b/` (la plus récente).

## 5. Résultats DPC de Simon
| Run | Version | 1res places | Classement général |
|---|---|---|---|
| 21/10 | W3 | 33 % | 51e |
| 24/10 | W3 | 67 % | 51e |
| 31/10 | W4 | 83 % | 15e (demande faible) |
| 2/11 | W4 | 56 % | **9e** (demande forte, monopole : 2009 billets à ~51 €) |
| 4/11 → 16/11 | W4 | 47-63 % | 11e, 9e, 8e, 13e, 12e, 11e |

Alois : W2 16e et 67e / 75, W3 46e et 43e / 75 ; W4 du 2/11 : 1751 / saison, 62 sièges, classement à demander.

## 6. Simulateur et comparaison équitable
- `presentations/P1/team_analysis.py` (à lancer depuis la racine avec `venv/Scripts/python.exe`) : 66 simulations (46 de Simon, 20 d'Alois), un monde OLS par simulation, prix du concurrent rejoués, demande Poisson. Les 4 algos (Simon W3/W4, Alois W3/W4) jouent les mêmes mondes.
- Résultat : chaque algo gagne sur les données de son auteur ; sur les 50 simulations jamais vues par le modèle de Simon : Alois W4 2169, Alois W3 2167, Simon W4 2063, Simon W3 1950. Même conclusion avec un poids égal par run (Alois W4 devant sur 6 runs sur 6).
- Le run d'Alois du 2/11 n'est pas encore dans la simulation.
- `presentations/P1/dpc_files.py` : lit les dates des CSV (`10_21_2026` mois_jour, `2_11_2026` jour_mois, et depuis le 18/11 `21_10_26` jour_mois_aa). Toujours l'utiliser pour lister les CSV.
- Simulateur W7 (`W7-Notebooks/w7_simulation.py`) : 259 compétitions, saisons jouées à la suite (apprentissage), concurrent réactif (k mesuré) ou rejoué ; parallélisé. Les concurrents réagissent à notre prix d'hier (19 copieurs exacts sur 217).
- `presentations/P1/make_figures.py` : génère les graphiques dans `presentations/P1/figures/`.
- Limites : dans `team_analysis.py` (et W6), le concurrent ne réagit pas à nos prix ; demande linéaire ; on simule notre revenu, pas celui du concurrent (le classement compare les deux).

## 7. Présentation #1 (3/11)
- Version principale : https://claude.ai/artifact/JzApbTBuTXZ6M8VqgNk59f (18 diapos, notes d'orateur détaillées).
- Version anglais simple (mêmes diapos, notes 2 fois plus courtes) : https://claude.ai/artifact/KfR6A4Tn3VbxUzG7RxpKei
- Fichiers locaux des diapos : dossier scratchpad de la session (`deck2/` et `deck3/`), non conservés ; l'export `.pptx` est dans `presentations/P1/`.
- Si la présentation est publique ou modifiée sur la page par quelqu'un d'autre, Claude ne peut plus la lire ni la publier : la repasser en privé.
- Reste à faire : diapo 13, classement d'Alois du 2/11 (« ranking pending ») dans les deux versions.
- Le bug scipy de la W3 (prix à 150 €) n'a jamais tourné sur DPC : on n'en parle pas dans les diapos.

## 8. Problèmes traités plus tard par le cours
- Optimisation sous contrainte de capacité (80 sièges) → 10/11 : fait (W6). Pyomo + HiGHS installés dans le venv, pour les vérifications hors ligne seulement (pas disponibles sur DPC). LAD inutilisable sur nos données (60 % de jours à 0 vente).
- Dossiers de cours : `W6/` (10/11), `W7/` (17/11). Fiches visuelles des cours : `C:\Users\VCB\Fiches_visuelles\cours\` (skill /visuel).
- Combiner modèle hors ligne et apprentissage en ligne (Bertsimas & Perakis) → 17/11 : fait (W7).
- Exploiter la rupture du concurrent → 15/12.
- Courbe cible adaptée au niveau de demande (regrouper les compétitions) → 5/01.

## 9. Git & Environnement
- **Branche** : `simon` (`origin/simon`).
- `.gitignore` : `venv/`, `dpc_env/`, `__pycache__/`, `*.csv` sauf `résultat_DPC/*.csv`, et les copies de la branche d'Alois.
- Environnement Python : `venv/Scripts/python.exe`.
- Avant de soumettre : tester `duopoly.py` + `duopoly_params.pkl` seuls dans un dossier vide (dernier test : 11,5 ms par appel en moyenne).
