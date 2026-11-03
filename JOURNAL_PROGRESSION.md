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
| W4 (27/10, pas encore soumise) | `duopoly.py` + `duopoly_params.pkl` | courbe de remplissage avec marges, OLS commun aux 16 compétitions (chargé depuis un .pkl), prévision ARIMA du prix concurrent, mode monopole basé sur le modèle |

## Résultats sur la plateforme DPC (`résultat_DPC/`)

| Run | Version | 1re place | Classement moyen | Revenu moyen / simulation | Ruptures | Concurrents |
|---|---|---|---|---|---|---|
| ≤ 16/10 (CSV1) | W1/W2 initiale | ? | ? | ≈ 174 100 $ | ? | SaffronDolphin, AttentiveOriole |
| 21/10 | W3 corrigée | 33 % | 1,67 | 149 106 $ | 23,2 % | RustlingNightingale, FlashyInchworm, DelicateMoose |
| 24/10 | W3 corrigée | 67 % | 1,33 | 58 762 $ | 1,0 % | ElectricButterfly, IndependentPorpoise, Winner_2023 |
| 31/10 | **W4** | **83 %** | 1,17 | 52 334 $ (classement général : **15e**) | 0 % | PompousRaven, PinkDuck, ElectricButterfly, ImpartialGoldfish, VersedJackdaw, UncoveredElephant |
| 2/11 | W4 | 56 % | 1,44 | 236 759 $ (classement général : **9e**) ; 2368 / saison, 57,8 sièges | 9,2 % | DaringVole, StoicCentipede, IndependentPorpoise, TopazCat, SageSloth, WhimsicalMarten, PinkDuck, IncredibleDogfish, ImposingCrocodile (18 compétitions) |

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

## Décisions et pistes
- Ligne de conduite : suivre le cours semaine par semaine ; garder les améliorations hors cours pour plus tard.
- Couvert plus tard par le cours : exploiter la rupture du concurrent (15/12), regrouper les compétitions par type de demande (5/01), approche d'apprentissage de Bertsimas & Perakis (17/11).
