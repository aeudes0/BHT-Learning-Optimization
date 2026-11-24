# Dynamic pricing competition
3 simulations per week: Monday, Wednesday and Saturday, at 1.30 

## Week 8 24/11 - 1/12
IMPORTANT:
Ranking is more stable as all the competition have the same demand model
so for the presentation we should not say that the algo improved because the ranking stabilised
the 3 other ppl are ranked 3, 8 and 14 so we're not that good
Since there are only 30 competitors left (lots of the other were bots/copycats), a rank lower than 15 is bad. 
Reminder: teacher's algo following demand curve is 15 
Dynaming programming has nothing to do with the pricing
WARNING: it can be very ressourceful to do the recursion to find the maximum

Try to cluster the previous data into different data to do the regression
because if we do it on all of the previous data then it's not really interesting

For the ols, if R² is too small (<.2) ignore it, if between .2 and .4 look at the coefficient (price and all)
if we wait 20 days before doing the DP, we are losing 20% of our time and maybe lot of capacity, so we need to be sure that during the rest of the time we will be able to make up for it in the remaining time

next presentation: 
mainly about optimization so must have a better demand model
we can use poisson if we want

Modification of duopoly:

reworked it a lot
added clustering on the previous data to create 3 ols fit models
*train_clustering.py*
first days -> demand prediction with pyomo
then it tries to interpolate the demand, to chose which ols model to follow
then each 20 days it dynamicly computes a new prediction to change the ols model
also implemented the creation of a feedback data file
lot of WIP tho

Algorithm tested with *simulate_analyze.py* giving me Mode=historical and copycat to have an idea if the model is viable or not
then tries to optimize the prices

## Week 7 17/11 - 23/11
Implemented "simulate_analyze.py" script that shows us the result of our algorithm and returns the ideal price value
Pyomo: watch out because it can take a lot of time
maybe not use it that much online to match the time criteria, even if it can be used on the app
but could be useful for the offline training
need to focus on how do our demand model works

December 8 presentation:
dynamic programming
optimized prices -> can mean all and nothing, study of how do we find the optimized prize
for instance if we have a good demand model
Duopoly changes:
integrated the 4-step Pyomo LP (glpk/cbc) and NLP (ipopt) demand-estimation and pricing logic alongside your tuned constants and pace heuristic. To stay within DPC time limits, we moved solver detection to import time, capped solves to three checkpoints per season with an 8.0-second budget and None/error fallbacks, and blended cached Pyomo prices (0.25) with your heuristic (0.75). We also scoped the Windows idaes binary path setup to os.name == "nt" and left out external .pkl target-curve loading for now.
## Week 6 11/11 - 16/11
The result are moved to perf.csv to achieve easier visualisation of the results
introduction to Pyomo, didn't implement a new duopoly yet


## Week 4-5
Ran the same script 2 weeks, to have more data on our model
did a presentation on 2/11, teacher said it was good, see the "presentation" folder for more
### 26/10 - 10/11

build randomized capacity utilization curves
build OLS-based demand model 
f(price, competitor_price) = demand
price optimization based on the estimated demand model
price prediction using time series analysis
price optimization based on the predicted competitor's price

better OLS than last week:
 OLS Regression Results                            
==============================================================================
Dep. Variable:                 demand   R-squared:                       0.133
Model:                            OLS   Adj. R-squared:                  0.131
Method:                 Least Squares   F-statistic:                     76.51
Date:                Fri, 30 Oct 2026   Prob (F-statistic):           1.23e-31
Time:                        16:34:32   Log-Likelihood:                -489.26
No. Observations:                1000   AIC:                             984.5
Df Residuals:                     997   BIC:                             999.2
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
price               -0.0045      0.001     -5.856      0.000      -0.006      -0.003
price_competitor     0.0099      0.001     10.617      0.000       0.008       0.012
intercept            0.1866      0.028      6.656      0.000       0.132       0.242
==============================================================================
Omnibus:                      628.005   Durbin-Watson:                   2.030
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6353.533
Skew:                           2.798   Prob(JB):                         0.00
Kurtosis:                      14.008   Cond. No.                         79.4
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
### Results
Simulation:
1- 31/10 
- 38/75
- 0.0% stockout frequency
- $411.411 average revenue
2- 02/11
- 39/75
- 43.8% stockout frequency
- $175.110 average revenue
3- 04/11
- 14/75
- 22.0% stockout frequency
- $133.367 average revenue
4- 07/11
- 16/75
- 24.3% stockout frequency
- $173.003 average revenue
5- 09/11
- 18/75
- 27.6% stockout frequency
- $180.102 average revenue
6- 11/11
- 16/75
- 25.2% stockout frequency
- $160.819 average revenue
## Week 3
### 20/10 - 26/10
Create git repo
Pricing optimization using OLS (Ordinary Least Squares)

### Results
First Simulation: 
- 46/75
- 0.4% stockout frequency
- $194.596 average revenue
Second simulation:
- 43/75
- 0.0% stockout frequency
- $81.600 average revenue
Third simulation
Bugged
### Conclusion
Mid algorithm, not that bad though

## Week 2
### 13/10 - 19/10
Data visualization on different notebooks, create a tendency curve to follow.

### Results
First Simulation: did not participate
Second simulation:
- 16/75
- 0.2% stockout frequency
- $295.923 average revenue

Third simulation
- 67/75
- 0% stockout frequency
- $117.125 average revenue

### Conclusion
Basic model, sometimes it works sometimes it doesn't, normal for the first weeks

## Week 1
### Absent
I did not participate at the first class, but it was mainly general information about the class and introduction to the DPC contest
