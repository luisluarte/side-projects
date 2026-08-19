# Experimental Methodologies

This document defines the statistical and procedural logic executed by the five primary R scripts within the `src/experiments/` module. Each script interrogates a specific hypothesis regarding the Cortico-Cerebellar Expansion-Compression Manifold (ECCM).

---

## 1. Topologoical Ablation Proof
**Script:** `src/experiments/experiment_ablation.R`

### Objective
To prove that the high-dimensional projection of the Cerebellar Granule Cell layer provides a statistically significant improvement over a flat linear architecture when mapping deep historical chronologies.

### Methodology & Mathematical Intuition
We evaluate a null hypothesis ($H_0$) that a model stripped of its Granule Cells (the Lesioned model) yields an equivalent log-likelihood deviance to the Intact model. Because the models execute dynamically over trials rather than holding static weights, standard BIC/AIC criteria are invalid. Instead, we use Curvature-Penalized Deviance:

$$ \text{Deviance}(\theta) = \sum_{t=1}^{T} -2 \ln p(RT_t, Ch_t \mid v^{(t)}(\theta), a(\theta)) + \left\lvert \frac{\sum_{t} (t - \bar{t})(D_t - \bar{D})}{\sum_t (t-\bar{t})^2} \right\rvert $$

### Code Mapping
```R
# Evaluate the MAP estimate for the Intact Model
dev_intact <- eval_eccm_intact(chain_intact[iters, ], p_data$Resp, p_data$F, p_data$RT)

# Evaluate the MAP estimate for the Lesioned Model
dev_lesion <- eval_eccm_lesioned(chain_lesion[iters, ], p_data$Resp, p_data$F, p_data$RT)

# Execute the paired non-parametric Wilcoxon test across subjects
wilcox.test(results$Deviance_Intact, results$Deviance_Lesion, paired = TRUE, alternative = "less")
```

---

## 2. Full Cohort Bayesian Parameter Estimation
**Script:** `src/experiments/experiment_full_cohort_fit.R`

### Objective
Scale the Metropolis-within-Gibbs sampling pipeline across the full N=128 human participant cohort using `doParallel` to establish definitive, convergent MAP parameter sets.

### Methodology & Mathematical Intuition
Rather than relying on cross-sectional pooled analysis, a hierarchical ideographic approach evaluates the parameters subject-by-subject. Parallel threading loops independent MCMC samplers.

### Code Mapping
```R
# Orchestration delegates to bayesian_fitting_orchestrator.R
results <- run_full_cohort_mcmc(dataset_path, out_csv_path, iters = 150)

# Statistically contrast the final cohort deviances
t.test(results$Deviance_Intact, results$Deviance_WSLS, paired=TRUE, alternative="less")
```

---

## 3. Metric Comparisons: PR-AUC and RT-RMSE
**Script:** `src/experiments/experiment_metric_comparison.R`

### Objective
Translate the abstract deviance metrics into human-interpretable performance bounds: Precision-Recall Area Under the Curve (PR-AUC) and Expected Reaction Time Root Mean Square Error (RT-RMSE).

### Methodology & Mathematical Intuition
Using the MAP parameters extracted in the prior experiment, we project the choice probabilities and expected RTs:

$$ P(Ch_t = 1) = \frac{1}{1 + e^{-\frac{2 a_t v_t}{c^2}}} $$
$$ E[RT_t] = t_{nd} + \frac{a_t}{v_t} \tanh(a_t v_t) $$

### Code Mapping
```R
# Project Expected Choice Probability using the C++ evaluation
prob_ch1 <- projected_v$prob_ch1

# Compute PR-AUC using the PRROC package
pr_curve <- pr.curve(scores.class0 = prob_ch1[p_data$Resp == 1],
                     scores.class1 = prob_ch1[p_data$Resp == 2], curve = TRUE)
prauc_intact[p_idx] <- pr_curve$auc.integral
```

---

## 4. Stratified Hyper-Alert State Tracking
**Script:** `src/experiments/experiment_stratified_tracking.R`

### Objective
Prove that the Cerebellum executes an immediate, massive non-linear re-mapping of its high-dimensional topological geometry when encountering a "volatile" pause, rapidly forcing Cortical predictions to $0.0$ to trigger a hyper-alert heuristic search state.

### Methodology & Mathematical Intuition
Pauses are stratified into:
1. **Group A (Stationary):** No underlying probability flip during the temporal break.
2. **Group B (Volatile):** The environment secretly reversed during the break.

We define Discrepancy as: $\Delta_{CC} = |Q_{CTX} - Q_{CB}|$

### Code Mapping
```R
# Detect sequence indices where the pause exceeds 35 seconds
large_pauses <- which(p_data$delta_t_s > 35)

# Calculate pre- and post-pause generative probabilities to classify Group A vs B
pre_prob <- mean(p_data$F[(t-10):(t-1)])
post_prob <- mean(p_data$F[(t):(t+9)])

# Extract the Q_CB and Q_CTX states from the topology extractor output
delta_cc_post <- abs(states$Q_CTX[t:(t+4)] - states$Q_CB[t:(t+4)])
```
