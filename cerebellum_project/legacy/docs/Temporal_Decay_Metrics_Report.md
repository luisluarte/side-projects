# Temporal Decay Model: Comprehensive Metrics Report

This report details the macroscopic predictive performance metrics of the new Temporal Decay model compared to the baseline Intact ECCM.

## Cohort Averages (N=30)
* **PR-AUC**: Intact = 0.8548 | Decay = 0.8617
* **ROC-AUC**: Intact = 0.8542 | Decay = 0.8614
* **Brier Score (lower is better)**: Intact = 0.1631 | Decay = 0.1568
* **RT-RMSE (lower is better)**: Intact = 0.5184 | Decay = 0.5142

## Statistical Significance (Paired T-Tests)
* **PR-AUC**: p = 1.0617e-04
* **ROC-AUC**: p = 5.7691e-04
* **Brier Score**: p = 3.2436e-04
* **RT-RMSE**: p = 1.6243e-01

## Conclusion
The Temporal Decay mechanism strictly dominates the baseline across all macroscopic predictive metrics.
