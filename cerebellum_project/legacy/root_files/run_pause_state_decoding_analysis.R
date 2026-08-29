# ==============================================================================
# IDIOGRAPHIC PAUSE-STATE DECODING & OBSERVABLE LOGISTIC REGRESSION
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(stats)
  library(pROC)
  library(PRROC)
  library(ggplot2)
  library(gridExtra)
})

cat("==============================================================================\n")
cat("STARTING IDIOGRAPHIC PAUSE-STATE DECODING ANALYSIS (128 PARTICIPANTS)\n")
cat("==============================================================================\n\n")

sourceCpp("ExactRModel.cpp")

dataset_path <- "../datasets/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

pop_matrix_path <- "idiographic_population_parameter_matrix.csv"
if (!file.exists(pop_matrix_path)) {
  stop("idiographic_population_parameter_matrix.csv not found! Run idiographic optimization first.")
}
df_pop <- read.csv(pop_matrix_path)

participants <- unique(dat_all[['participant_id']])
N_sub <- length(participants)

# Param names in theta
param_names <- c("p_ws_base", "p_ls_base", "w_mag_curr", "w_mag_alt", "alpha_q", 
                 "w_streak", "w_purkinje_inh", "tau_kinematic", "beta_post_err", "kappa_entropy")

# Macroscopic Pause Threshold (seconds)
PAUSE_THRESHOLD_SEC <- 10.0

stacked_df_list <- list()
representative_sub_df <- NULL

for (s in 1:N_sub) {
  p_id <- participants[s]
  sub_df <- dat_all[dat_all[['participant_id']] == p_id, ]
  resp <- as.numeric(sub_df[['Resp']])
  out <- as.numeric(sub_df[['F']])
  m1 <- as.numeric(sub_df[['Bd1']])
  m2 <- as.numeric(sub_df[['Bd2']])
  rt <- as.numeric(sub_df[['RT']])
  ttp <- as.numeric(sub_df[['ttp']]) / 1000.0 # convert to seconds
  N_t <- length(resp)
  
  th_s <- as.numeric(df_pop[df_pop$participant_id == p_id, param_names])
  
  # Forward pass with converged idiographic theta
  res <- run_exact_r_simulation_cpp(resp, out, m1, m2, rt, th_s)
  
  val_t <- as.numeric(res$Value_Traj)
  unc_t <- as.numeric(res$Uncertainty_Traj)
  snorm_t <- as.numeric(res$State_Norm_Traj)
  
  # Compute ITI Delta t
  delta_t <- c(0, diff(ttp))
  pause_recovery <- ifelse(delta_t >= PAUSE_THRESHOLD_SEC, 1, 0)
  
  sub_trials_df <- data.frame(
    participant_id = p_id,
    trial_idx = 1:N_t,
    Resp = resp,
    Outcome = out,
    RT = rt,
    Delta_t = delta_t,
    Pause_Recovery = pause_recovery,
    Value = val_t,
    Uncertainty = unc_t,
    State_Norm = snorm_t
  )
  
  stacked_df_list[[s]] <- sub_trials_df
  if (s == 1) {
    representative_sub_df <- sub_trials_df
  }
}

df_all_trials <- do.call(rbind, stacked_df_list)

cat(sprintf("Total trials analyzed: %d | Total Pause-Recovery events detected: %d (%.2f%%)\n\n",
            nrow(df_all_trials), sum(df_all_trials$Pause_Recovery), 
            100 * mean(df_all_trials$Pause_Recovery)))

# ==============================================================================
# LOGISTIC REGRESSION GLM FIT
# ==============================================================================
cat("Fitting Binomial Generalized Linear Model (Logistic Regression Decoder)...\n")

glm_fit <- glm(Pause_Recovery ~ Uncertainty + State_Norm + Value, 
               data = df_all_trials, 
               family = binomial(link = "logit"))

summary_glm <- summary(glm_fit)
print(summary_glm)

# Compute Odds Ratios and 95% Confidence Intervals
coef_mat <- summary_glm$coefficients
odds_ratios <- exp(coef_mat[, "Estimate"])
ci_low <- exp(coef_mat[, "Estimate"] - 1.96 * coef_mat[, "Std. Error"])
ci_high <- exp(coef_mat[, "Estimate"] + 1.96 * coef_mat[, "Std. Error"])

df_glm_results <- data.frame(
  Predictor = rownames(coef_mat),
  Estimate = coef_mat[, "Estimate"],
  Std_Error = coef_mat[, "Std. Error"],
  z_value = coef_mat[, "z value"],
  p_value = coef_mat[, "Pr(>|z|)"],
  Odds_Ratio = odds_ratios,
  CI_2.5 = ci_low,
  CI_97.5 = ci_high,
  stringsAsFactors = FALSE
)

write.csv(df_glm_results, "pause_state_logistic_regression_results.csv", row.names = FALSE)
cat("Saved pause_state_logistic_regression_results.csv\n\n")

# ==============================================================================
# CROSS-VALIDATED ROC-AUC EVALUATION
# ==============================================================================
cat("Computing 10-Fold Cross-Validated Decoding ROC-AUC & PR-AUC...\n")

set.seed(42)
folds <- sample(rep(1:10, length.out = nrow(df_all_trials)))
cv_preds <- numeric(nrow(df_all_trials))

for (f in 1:10) {
  train_data <- df_all_trials[folds != f, ]
  test_data  <- df_all_trials[folds == f, ]
  
  fit_f <- glm(Pause_Recovery ~ Uncertainty + State_Norm + Value, 
               data = train_data, 
               family = binomial(link = "logit"))
  cv_preds[folds == f] <- predict(fit_f, newdata = test_data, type = "response")
}

roc_obj <- pROC::auc(df_all_trials$Pause_Recovery, cv_preds)
pr_obj <- pr.curve(scores.class0 = cv_preds[df_all_trials$Pause_Recovery == 1],
                   scores.class1 = cv_preds[df_all_trials$Pause_Recovery == 0], curve = FALSE)

cat(sprintf("Cross-Validated Pause Decoding ROC-AUC: %.4f | PR-AUC: %.4f\n\n",
            as.numeric(roc_obj), pr_obj[['auc.integral']]))

# ==============================================================================
# MULTI-PANEL TIME-SERIES VISUALIZATIONS
# ==============================================================================
cat("Generating Multi-Panel Time-Series Visualization Plots...\n")

sub1 <- representative_sub_df[1:120, ]

p1 <- ggplot(sub1, aes(x = trial_idx, y = Delta_t)) +
  geom_line(color = "#333333", linewidth = 0.8) +
  geom_point(aes(color = factor(Pause_Recovery)), size = 2.5) +
  geom_hline(yintercept = PAUSE_THRESHOLD_SEC, linetype = "dashed", color = "darkred", linewidth = 0.8) +
  scale_color_manual(values = c("0" = "#555555", "1" = "#e74c3c"), 
                     labels = c("Standard Trial", "Pause Recovery Event"),
                     name = "Trial State") +
  theme_minimal(base_size = 12) +
  labs(title = "A. Empirical Inter-Trial Interval (ITI) Spikes & Macroscopic Pauses",
       y = "Elapsed ITI Delta t (s)", x = "Trial Index") +
  theme(legend.position = "top", plot.title = element_text(face = "bold", color = "#003366"))

p2 <- ggplot(sub1, aes(x = trial_idx)) +
  geom_line(aes(y = Uncertainty, color = "Uncertainty U_t"), linewidth = 1.1) +
  geom_line(aes(y = State_Norm / 2.5, color = "Normalized State Norm ||z_GC|| / 2.5"), linewidth = 1.1) +
  scale_color_manual(values = c("Uncertainty U_t" = "#e67e22", 
                                "Normalized State Norm ||z_GC|| / 2.5" = "#2980b9"),
                     name = "Reservoir Observable") +
  theme_minimal(base_size = 12) +
  labs(title = "B. Continuous Cortico-Cerebellar Observables during Task Resumption",
       y = "Observable Amplitude", x = "Trial Index") +
  theme(legend.position = "top", plot.title = element_text(face = "bold", color = "#003366"))

p_combined <- grid.arrange(p1, p2, ncol = 1)
ggsave("pause_state_decoding_timeseries.png", plot = p_combined, width = 10.0, height = 7.0, dpi = 300)
cat("Saved pause_state_decoding_timeseries.png\n")

cat("\n==============================================================================\n")
cat("PAUSE-STATE DECODING ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
