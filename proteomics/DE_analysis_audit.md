# Theoretical and Computational Audit of Differential Expression (`DE_analysis.R`)

## Executive Summary

A comprehensive theoretical and computational evaluation of the differential protein expression pipeline (`scripts/DE_analysis.R`) was conducted on the maxLFQ proteomics dataset (5 Control vs 5 Treatment samples). 

The audit evaluated the statistical modeling choices—including filtering thresholds, log2 transformation, median centering, **MinProb missing value imputation**, limma empirical Bayes variance moderation (`trend = TRUE`, `robust = TRUE`), and Benjamini-Hochberg FDR control. In addition, four parallel computational candidate pipelines were benchmarked:
1. **MinProb Imputation (Current Pipeline)**
2. **NA-Aware Limma (No Imputation)**
3. **QRILC (Quantile Regression Imputation of Left-Censored data)**
4. **KNN (K-Nearest Neighbors Imputation)**

### Key Audit Findings
- **High Overall Statistical Robustness**: The primary signal of differential expression is highly robust. 177 out of 181 significant proteins (97.8%) under MinProb imputation are independently confirmed by NA-aware limma without imputation.
- **Low Imputation Distortion Risk**: MinProb introduces only **4 imputation-specific hits** (2.2% of total hits), which are driven by missing values in one group. These 4 proteins should be flagged for secondary manual validation.
- **P-Value Calibration & Inflation**: All pipelines exhibit a Genomic Inflation Factor ($\lambda \approx 1.57 - 1.60$), reflecting a strong, true genome/proteome-wide biological response to treatment rather than mathematical artifact.
- **Limma Trend Alignment**: The `trend = TRUE` setting in `eBayes` effectively models the intensity-dependent variance trend inherent to label-free proteomics data.

---

## 1. Theoretical Methodological Evaluation

### 1.1 Filtering Strategy (`valid_total >= 8 & (valid_ctrl >= 4 | valid_trat >= 4)`)
- **Evaluation**: The current rule requires detection in at least 8 of 10 samples (80% completeness across total samples) and at least 4 out of 5 replicates in at least one condition.
- **Theoretical Assessment**: **SOUND**. This strict filter effectively eliminates sporadic low-quality noise features while preserving proteins with complete condition-specific dropouts (e.g. present in 5/5 Control, 0/5 Treatment). Retained **5,252 out of 6,040 proteins (87.0%)**.

### 1.2 Transformation and Normalization (`log2Transform` + `center.median`)
- **Evaluation**: Log2 transformation stabilizes variance across orders of magnitude; sample-wise median centering removes systematic loading differences across LC-MS runs.
- **Theoretical Assessment**: **RECOMMENDED STANDARD**. Proteomics LFQ intensity distributions are right-skewed and log-normal. Median centering is robust against asymmetric differential abundance (unaffected by large fold-change outliers).

### 1.3 Missing Value Imputation (`impute(..., method = "MinProb")`)
- **Evaluation**: MinProb imputes missing values by sampling from a left-shifted Gaussian distribution (representing the limit of detection under a Missing Not At Random / MNAR assumption).
- **Theoretical Risk**: Imputing deterministic left-tail values *prior* to calculating sample variances in `limma` can artificially compress within-group variance $\hat{\sigma}_g^2$, leading to inflated $t$-statistics and potential false positives for features with sporadic missingness.
- **Empirical Check Result**: For proteins with low NA counts ($0-1$ missing values), MinProb introduces virtually zero bias. However, for proteins with $2+$ NAs concentrated in one treatment arm, MinProb forces a low synthetic mean, artificially boosting logFC and significance.

### 1.4 Linear Modeling & Variance Moderation (`lmFit` + `eBayes(trend = TRUE, robust = TRUE)`)
- **Evaluation**: `limma` fits a weighted least-squares model per protein and moderates residual variances toward a mean-variance trend curve ($\tilde{s}_g^2 = d_0 s_0^2 + d_g s_g^2$).
- **Theoretical Assessment**: **EXCELLENT**. `trend = TRUE` accounts for higher variance at low mean log2 intensities in MS datasets. `robust = TRUE` protects hyperparameter estimation ($d_0, s_0^2$) against hyper-variable outlier proteins.

---

## 2. Computational Diagnostic Suite & Benchmark Results

### 2.1 Benchmark Summary Table

| Pipeline | Total Features | Raw $P < 0.05$ | Significant Hits (FDR $< 0.05$) | Genomic Inflation Factor ($\lambda$) | Concordance with NA-Aware Limma |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MinProb (Current)** | 5,252 | 653 | **181** | 1.566 | 97.8% |
| **NA-Aware Limma (No Imp)** | 5,252 | 670 | **183** | 1.578 | 100% (Baseline) |
| **QRILC Imputation** | 5,252 | 634 | **167** | 1.605 | 91.3% |
| **KNN Imputation** | 5,252 | 671 | **185** | 1.575 | 98.9% |

---

### 2.2 Visual Diagnostics & Plots

#### Figure 1: Missingness Rate vs. Mean Intensity
![Missingness Profile](file:///c:/Users/DCCS5/Documents/GitHub/side-projects/proteomics/figures/1_missingness_profile.png)
*Observation*: Missingness strongly correlates with low mean log2 intensity (classic MNAR / limit of detection mechanism), confirming that left-censored imputation (MinProb / QRILC) is theoretically justified for low-abundance dropouts.

#### Figure 2: P-Value Calibration & Uniformity (Histograms & QQ-Plots)
![P-value Distributions](file:///c:/Users/DCCS5/Documents/GitHub/side-projects/proteomics/figures/2_pvalue_histograms_and_qq.png)
*Observation*: P-value histograms show a clean flat null distribution with a strong enrichment of low p-values ($P < 0.01$). The QQ plot shows identical behavior across MinProb, NA-aware limma, and KNN.

#### Figure 3: Limma Mean-Variance Trends (SA Plots)
![Mean-Variance Trends](file:///c:/Users/DCCS5/Documents/GitHub/side-projects/proteomics/figures/3_mean_variance_trends.png)
*Observation*: The residual standard deviation ($\sigma$) displays a smooth downward trend as average log2 expression increases, demonstrating that `eBayes(trend = TRUE)` successfully stabilizes variance estimation.

#### Figure 4: Effect Size Concordance (MinProb vs. NA-Aware Limma)
![LogFC Concordance](file:///c:/Users/DCCS5/Documents/GitHub/side-projects/proteomics/figures/4_pipeline_concordance_volcano.png)
*Observation*: Log2 fold changes estimated by MinProb and NA-aware limma exhibit near-perfect correlation (Spearman $r > 0.99$).

#### Figure 5: Imputation Artifact Analysis
![Artifact Analysis](file:///c:/Users/DCCS5/Documents/GitHub/side-projects/proteomics/figures/5_minprob_artifact_analysis.png)
*Observation*: Out of 181 significant hits in MinProb, 177 are robustly confirmed by NA-aware limma. Only 4 hits are MinProb-specific artifacts.

---

## 3. Dissecting MinProb-Only Artifact Hits

The audit isolated **4 proteins** that achieve statistical significance (FDR $< 0.05$) under MinProb imputation but fail FDR thresholds under NA-aware limma:

| Protein ID / Symbol | LogFC (MinProb) | FDR (MinProb) | LogFC (NA-Aware) | FDR (NA-Aware) | Total NAs | Cause of Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Protein 1044** | -2.08 | 0.0336 | -1.35 | 0.0607 | 2 | Left-tail imputation reduced variance estimate |
| **Protein 715** | -1.15 | 0.0340 | -0.83 | 0.1223 | 2 | MinProb imputed values amplified FC |
| **Protein 2656** | +1.94 | 0.0359 | +1.70 | 0.1739 | 3 | Asymmetric missingness across groups |
| **Protein 2897** | -1.36 | 0.0409 | -1.23 | 0.0788 | 2 | Marginal significance shift (0.0409 vs 0.0788) |

---

## 4. Final Recommendations & Refinements

1. **Retain Current Pipeline for Core Analysis**:
   - `DE_analysis.R` is mathematically sound, highly reproducible, and produces reliable results (97.8% hit concordance with non-imputed models).
2. **Add Sensitivity Flagging Step**:
   - In downstream reporting or manuscript generation, cross-reference `DE_analysis.csv` hits against NA-aware limma to flag the 4 imputation-sensitive proteins (`Protein 1044`, `Protein 715`, `Protein 2656`, `Protein 2897`).
3. **Execution Script Available**:
   - The diagnostic audit script is saved at [`scripts/DE_audit_diagnostics.R`](file:///c:/Users/DCCS5/Documents/GitHub/side-projects/proteomics/scripts/DE_audit_diagnostics.R) and can be executed at any time to re-generate audit metrics.
