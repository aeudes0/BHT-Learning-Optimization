# Dynamic pricing competition
3 simulations per week: Monday, Wednesday and Saturday, at 2.30 

## Week 4
### 26/10 - 2/11

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
