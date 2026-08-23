# Hierarchical Evaluation Results

## Phase 1: PSIS-LOO (Out-of-Sample Supremacy)
- **M6**: Delta elpd = 0.00
- **M18**: Delta elpd = -177.80
- **M19**: Delta elpd = -1041.24
- **M20**: Delta elpd = -785.77

## Phase 2: ZIB Regression (Delta Recall)
```
 Family: zero_inflated_beta 
  Links: mu = logit; zi = logit 
Formula: delta_recall ~ model + (1 | subject) 
         zi ~ model
   Data: df_phase2 (Number of observations: 180) 
  Draws: 2 chains, each with iter = 1000; warmup = 500; thin = 1;
         total post-warmup draws = 1000

Multilevel Hyperparameters:
~subject (Number of levels: 30) 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
sd(Intercept)     0.00      0.00     0.00     0.00 1.00      850      362

Regression Coefficients:
             Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept        0.00      0.00    -0.01     0.01 1.00      991      620
zi_Intercept   -10.04      6.53   -28.44    -2.93 1.01      257      126
modelM19        -0.00      0.01    -0.01     0.01 1.00     1029      817
modelM20        -0.00      0.01    -0.01     0.01 1.00      973      818
modelM6         -0.00      0.01    -0.01     0.01 1.00     1179      630
modelQL         -0.00      0.01    -0.01     0.01 1.00     1010      606
modelWSLS        0.00      0.01    -0.01     0.01 1.00     1136      732
zi_modelM19      0.40      8.88   -17.40    20.73 1.02      199      206
zi_modelM20     -0.14      8.82   -18.30    18.77 1.01      271      142
zi_modelM6       0.20      8.60   -16.31    18.73 1.00      221      194
zi_modelQL       0.14      8.75   -18.09    19.12 1.01      292      241
zi_modelWSLS     0.31      9.37   -19.10    19.69 1.01      214      214

Further Distributional Parameters:
    Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
phi  8672.11    896.15  7010.41 10389.51 1.00     1298      681

Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1).
```

## Phase 3: Beta Regression (Epistemic Calibration)
```
 Family: beta 
  Links: mu = logit 
Formula: bs ~ model * decile + (1 | subject) 
   Data: df_phase3 (Number of observations: 1800) 
  Draws: 2 chains, each with iter = 1000; warmup = 500; thin = 1;
         total post-warmup draws = 1000

Multilevel Hyperparameters:
~subject (Number of levels: 30) 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
sd(Intercept)     0.00      0.00     0.00     0.00 1.01      819      515

Regression Coefficients:
                 Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept            0.00      0.00    -0.00     0.00 1.00     1209      547
modelM19            -0.00      0.00    -0.00     0.00 1.00     1176      704
modelM20            -0.00      0.00    -0.00     0.00 1.01     1136      634
modelM6             -0.00      0.00    -0.00     0.00 1.00     1173      764
modelQL             -0.00      0.00    -0.00     0.00 1.00     1183      689
modelWSLS           -0.00      0.00    -0.00     0.00 1.00     1081      670
decile              -0.00      0.00    -0.00     0.00 1.00     1279      565
modelM19:decile      0.00      0.00    -0.00     0.00 1.00     1205      679
modelM20:decile     -0.00      0.00    -0.00     0.00 1.01     1049      867
modelM6:decile       0.00      0.00    -0.00     0.00 1.00     1156      865
modelQL:decile       0.00      0.00    -0.00     0.00 1.01     1185      676
modelWSLS:decile     0.00      0.00    -0.00     0.00 1.00     1117      587

Further Distributional Parameters:
    Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
phi 89389.49   2831.05 84149.42 95222.42 1.01     1596      534

Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1).
```

## Phase 4: Gamma Regression (Kinetic Alignment W1)
```
 Family: gamma 
  Links: mu = log 
Formula: w1 ~ model * decile + (1 | subject) 
   Data: df_phase4 (Number of observations: 1200) 
  Draws: 2 chains, each with iter = 1000; warmup = 500; thin = 1;
         total post-warmup draws = 1000

Multilevel Hyperparameters:
~subject (Number of levels: 30) 
              Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
sd(Intercept)     0.32      0.05     0.24     0.42 1.04       60      199

Regression Coefficients:
                Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept          -0.83      0.07    -0.97    -0.67 1.01      161      338
modelM19           -0.03      0.07    -0.16     0.10 1.01      388      526
modelM20            0.04      0.07    -0.09     0.17 1.01      388      575
modelM6            -0.08      0.07    -0.22     0.05 1.01      421      619
decile              0.03      0.01     0.01     0.04 1.01      307      570
modelM19:decile     0.01      0.01    -0.01     0.03 1.02      345      454
modelM20:decile    -0.00      0.01    -0.02     0.02 1.01      337      515
modelM6:decile      0.01      0.01    -0.01     0.03 1.01      380      514

Further Distributional Parameters:
      Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
shape     5.55      0.22     5.12     6.02 1.00     1139      844

Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1).
```
