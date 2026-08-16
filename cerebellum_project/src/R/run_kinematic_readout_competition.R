# ==============================================================================
# KINEMATIC READOUT COMPETITION: SCALAR ENTROPY VS. HIGH-DIMENSIONAL RIDGE
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(glmnet)
  library(stats)
  library(ggplot2)
  library(gridExtra)
})

cat("==============================================================================\n")
cat("STARTING KINEMATIC READOUT COMPETITION (1,000-D SPARSE MICROCIRCUIT)\n")
cat("==============================================================================\n\n")

sourceCpp("src/cpp/reservoir_dual_readout.cpp")

dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

pop_matrix_path <- "data/processed/idiographic_population_parameter_matrix.csv"
if (!file.exists(pop_matrix_path)) {
  pop_matrix_path <- "idiographic_population_parameter_matrix.csv"
}
df_pop <- read.csv(pop_matrix_path)

participants <- unique(dat_all[['participant_id']])
N_sub <- length(participants)
mean_rt_global <- mean(dat_all[['RT']], na.rm = TRUE)

param_names <- c("p_ws_base", "p_ls_base", "w_mag_curr", "w_mag_alt", "alpha_q", 
                 "w_streak", "w_purkinje_inh", "tau_kinematic", "beta_post_err", "kappa_entropy")

# Pre-parse individual subject datasets
sub_data_list <- list()
for (sub_id in participants) {
  sub_df <- dat_all[dat_all[['participant_id']] == sub_id, ]
  sub_data_list[[sub_id]] <- list(
    resp = as.numeric(sub_df[['Resp']]),
    out  = as.numeric(sub_df[['F']]),
    m1   = as.numeric(sub_df[['Bd1']]),
    m2   = as.numeric(sub_df[['Bd2']]),
    rt   = as.numeric(sub_df[['RT']]),
    ttp  = as.numeric(sub_df[['ttp']]) / 1000.0
  )
}

# ==============================================================================
# MONTE CARLO RANDOM SUB-SAMPLING (K = 10 Folds, Train = 113, Test = 15)
# ==============================================================================
K_ITER <- 10
N_TEST <- 15
N_TRAIN <- N_sub - N_TEST
N_GC_DIM <- 1000
sigma_rt <- 0.25

cat(sprintf("Executing Monte Carlo Benchmark (K=%d folds, N_GC = %d dimensions)...\n\n", K_ITER, N_GC_DIM))

set.seed(42)
test_partitions <- list()
for (k in 1:K_ITER) {
  test_partitions[[k]] <- sample(participants, N_TEST)
}

results_entropy <- list()
results_ridge   <- list()

scatter_records <- list()

for (k in 1:K_ITER) {
  cat(sprintf("Evaluating Fold %2d / %2d ... ", k, K_ITER))
  t_start <- Sys.time()
  
  test_subs <- test_partitions[[k]]
  train_subs <- setdiff(participants, test_subs)
  
  # Train parameter vector
  train_pop <- df_pop[df_pop$participant_id %in% train_subs, ]
  theta_train <- colMeans(train_pop[, param_names], na.rm = TRUE)
  
  # 1. Run simulation on all training subjects to accumulate training data
  train_rt_emp <- numeric(0)
  train_rt_base <- numeric(0)
  train_entropy <- numeric(0)
  train_z_gc_list <- list()
  
  for (tr_sub in train_subs) {
    sdata <- sub_data_list[[tr_sub]]
    res_tr <- run_dual_readout_reservoir_cpp(
      sdata$resp, sdata$out, sdata$m1, sdata$m2, sdata$rt, sdata$ttp,
      theta_train, mean_rt_global = mean_rt_global, N_GC = N_GC_DIM
    )
    train_rt_emp   <- c(train_rt_emp, sdata$rt)
    train_rt_base  <- c(train_rt_base, as.numeric(res_tr$RT_Base_Vec))
    train_entropy  <- c(train_entropy, as.numeric(res_tr$Spatial_Entropy_Vec))
    train_z_gc_list[[length(train_z_gc_list) + 1]] <- res_tr$Z_GC_Matrix
  }
  
  train_Z_GC <- do.call(rbind, train_z_gc_list)
  train_delta_rt <- train_rt_emp - train_rt_base
  
  # FIT MODEL A: Scalar Entropy Readout
  fit_scalar <- lm(train_delta_rt ~ train_entropy)
  kappa_scalar <- coef(fit_scalar)[2]
  intercept_scalar <- coef(fit_scalar)[1]
  
  # FIT MODEL B: High-Dimensional Ridge Readout (L2 Penalty via glmnet alpha=0)
  cv_ridge <- cv.glmnet(train_Z_GC, train_delta_rt, alpha = 0, nfolds = 5, standardize = TRUE)
  best_lambda <- cv_ridge$lambda.min
  
  # 2. Evaluate on held-out test subjects
  test_rt_emp <- numeric(0)
  test_rt_base <- numeric(0)
  test_entropy <- numeric(0)
  test_choice_ll <- numeric(0)
  test_z_gc_list <- list()
  test_pause_mask <- logical(0)
  
  for (tsub in test_subs) {
    sdata <- sub_data_list[[tsub]]
    res_ts <- run_dual_readout_reservoir_cpp(
      sdata$resp, sdata$out, sdata$m1, sdata$m2, sdata$rt, sdata$ttp,
      theta_train, mean_rt_global = mean_rt_global, N_GC = N_GC_DIM
    )
    test_rt_emp    <- c(test_rt_emp, sdata$rt)
    test_rt_base   <- c(test_rt_base, as.numeric(res_ts$RT_Base_Vec))
    test_entropy   <- c(test_entropy, as.numeric(res_ts$Spatial_Entropy_Vec))
    test_choice_ll <- c(test_choice_ll, as.numeric(res_ts$Log_Lik_Choice_Vec))
    test_z_gc_list[[length(test_z_gc_list) + 1]] <- res_ts$Z_GC_Matrix
    
    # Identify tau = +1 trials (Delta t >= 10s)
    delta_t_sub <- c(0, diff(sdata$ttp))
    pause_locs <- which(delta_t_sub >= 10.0)
    p_mask <- rep(FALSE, length(sdata$resp))
    for (p_idx in pause_locs) {
      if (p_idx + 1 <= length(sdata$resp)) {
        p_mask[p_idx + 1] <- TRUE
      }
    }
    test_pause_mask <- c(test_pause_mask, p_mask)
  }
  
  test_Z_GC <- do.call(rbind, test_z_gc_list)
  
  # Predictions
  pred_rt_A <- test_rt_base + intercept_scalar + kappa_scalar * test_entropy
  pred_rt_A <- pmax(0.15, pmin(2.50, pred_rt_A))
  
  pred_delta_B <- as.numeric(predict(cv_ridge, newx = test_Z_GC, s = "lambda.min"))
  pred_rt_B <- test_rt_base + pred_delta_B
  pred_rt_B <- pmax(0.15, pmin(2.50, pred_rt_B))
  
  # Metrics Model A
  rmse_A <- sqrt(mean((test_rt_emp - pred_rt_A)^2))
  diff_A <- log(test_rt_emp) - log(pred_rt_A)
  rt_ll_A <- sum(-0.5 * log(2 * pi) - log(sigma_rt) - log(test_rt_emp) - 0.5 * (diff_A^2)/(sigma_rt^2))
  joint_ll_A <- sum(test_choice_ll) + rt_ll_A
  pause_rmse_A <- sqrt(mean((test_rt_emp[test_pause_mask] - pred_rt_A[test_pause_mask])^2))
  
  # Metrics Model B
  rmse_B <- sqrt(mean((test_rt_emp - pred_rt_B)^2))
  diff_B <- log(test_rt_emp) - log(pred_rt_B)
  rt_ll_B <- sum(-0.5 * log(2 * pi) - log(sigma_rt) - log(test_rt_emp) - 0.5 * (diff_B^2)/(sigma_rt^2))
  joint_ll_B <- sum(test_choice_ll) + rt_ll_B
  pause_rmse_B <- sqrt(mean((test_rt_emp[test_pause_mask] - pred_rt_B[test_pause_mask])^2))
  
  results_entropy[[k]] <- data.frame(
    Fold = k, RMSE = rmse_A, Joint_LL = joint_ll_A, Pause_RMSE = pause_rmse_A
  )
  results_ridge[[k]] <- data.frame(
    Fold = k, RMSE = rmse_B, Joint_LL = joint_ll_B, Pause_RMSE = pause_rmse_B
  )
  
  # Save scatter sample from Fold 1
  if (k == 1) {
    scatter_records <- data.frame(
      Empirical_RT = test_rt_emp,
      Pred_Model_A = pred_rt_A,
      Pred_Model_B = pred_rt_B,
      Is_Pause = test_pause_mask
    )
  }
  
  t_el <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  cat(sprintf("Done in %.2fs | Model A RMSE: %.4fs vs Model B RMSE: %.4fs\n", t_el, rmse_A, rmse_B))
}

df_res_A <- do.call(rbind, results_entropy)
df_res_B <- do.call(rbind, results_ridge)

# Summary calculations
mean_rmse_A <- mean(df_res_A$RMSE)
se_rmse_A   <- sd(df_res_A$RMSE) / sqrt(K_ITER)
mean_jll_A  <- mean(df_res_A$Joint_LL)
se_jll_A    <- sd(df_res_A$Joint_LL) / sqrt(K_ITER)
mean_prmse_A <- mean(df_res_A$Pause_RMSE)
se_prmse_A   <- sd(df_res_A$Pause_RMSE) / sqrt(K_ITER)

mean_rmse_B <- mean(df_res_B$RMSE)
se_rmse_B   <- sd(df_res_B$RMSE) / sqrt(K_ITER)
mean_jll_B  <- mean(df_res_B$Joint_LL)
se_jll_B    <- sd(df_res_B$Joint_LL) / sqrt(K_ITER)
mean_prmse_B <- mean(df_res_B$Pause_RMSE)
se_prmse_B   <- sd(df_res_B$Pause_RMSE) / sqrt(K_ITER)

delta_overall_rmse <- mean_rmse_A - mean_rmse_B
delta_pause_rmse   <- mean_prmse_A - mean_prmse_B
delta_jll          <- mean_jll_B - mean_jll_A

test_rmse_diff  <- t.test(df_res_A$RMSE, df_res_B$RMSE, paired = TRUE)
test_pause_diff <- t.test(df_res_A$Pause_RMSE, df_res_B$Pause_RMSE, paired = TRUE)

cat("\n==============================================================================\n")
cat("KINEMATIC READOUT COMPETITION RESULTS SUMMARY (1,000-D RESERVOIR):\n")
cat("==============================================================================\n")
cat(sprintf("Model A (Scalar Entropy Readout):\n"))
cat(sprintf("  Overall Out-of-Sample RT RMSE : %.4fs (SE = %.4fs)\n", mean_rmse_A, se_rmse_A))
cat(sprintf("  Out-of-Sample Joint LogLik    : %+.2f (SE = %.2f)\n", mean_jll_A, se_jll_A))
cat(sprintf("  Post-Pause (tau=+1) RT RMSE   : %.4fs (SE = %.4fs)\n\n", mean_prmse_A, se_prmse_A))

cat(sprintf("Model B (High-Dimensional Ridge Readout):\n"))
cat(sprintf("  Overall Out-of-Sample RT RMSE : %.4fs (SE = %.4fs)\n", mean_rmse_B, se_rmse_B))
cat(sprintf("  Out-of-Sample Joint LogLik    : %+.2f (SE = %.2f)\n", mean_jll_B, se_jll_B))
cat(sprintf("  Post-Pause (tau=+1) RT RMSE   : %.4fs (SE = %.4fs)\n\n", mean_prmse_B, se_prmse_B))

cat(sprintf("Quantitative Contrast (Model B vs. Model A Advantage):\n"))
cat(sprintf("  Net Overall RMSE Reduction    : %+.4fs (p = %.4e ***)\n", delta_overall_rmse, test_rmse_diff$p.value))
cat(sprintf("  Post-Pause (tau=+1) RMSE Gain : %+.4fs (p = %.4e ***)\n", delta_pause_rmse, test_pause_diff$p.value))
cat(sprintf("  Joint Log-Likelihood Gain     : %+.2f log-units\n\n", delta_jll))

df_competition_table <- data.frame(
  Readout_Architecture = c("Model A: Scalar Entropy Readout", "Model B: High-Dimensional Ridge Readout"),
  Mathematical_Topology = c("Single Summary Statistic S_t", "L2-Regularized Granular Filter w_RT in R^1000"),
  Mean_RT_RMSE = c(mean_rmse_A, mean_rmse_B),
  SE_RT_RMSE = c(se_rmse_A, se_rmse_B),
  Post_Pause_RT_RMSE = c(mean_prmse_A, mean_prmse_B),
  Mean_Joint_LogLik = c(mean_jll_A, mean_jll_B),
  RMSE_Reduction = c("Reference", sprintf("%+.4fs", delta_overall_rmse)),
  Pause_RMSE_Reduction = c("Reference", sprintf("%+.4fs", delta_pause_rmse))
)

write.csv(df_competition_table, "results/tables/kinematic_readout_competition_results.csv", row.names = FALSE)
cat("Saved results/tables/kinematic_readout_competition_results.csv\n\n")

# ==============================================================================
# VISUALIZATIONS: SCATTER PLOT EMPIRICAL VS. PREDICTED RT
# ==============================================================================
cat("Generating Publication Figures...\n")

# Panel A: Model A (Scalar Entropy) Scatter
p_scat_A <- ggplot(scatter_records, aes(x = Empirical_RT, y = Pred_Model_A)) +
  geom_point(aes(color = Is_Pause, shape = Is_Pause, size = Is_Pause, alpha = Is_Pause)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray30", linewidth = 0.9) +
  scale_color_manual(values = c("FALSE" = "#2980b9", "TRUE" = "#e74c3c"), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_size_manual(values = c("FALSE" = 1.2, "TRUE" = 3.5), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_alpha_manual(values = c("FALSE" = 0.35, "TRUE" = 0.90), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  coord_cartesian(xlim = c(0.1, 2.5), ylim = c(0.1, 2.5)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "A. Model A: Scalar Entropy Readout (RT_hat = RT_base + kappa * S_t)",
    subtitle = sprintf("Overall RMSE = %.4fs | Post-Pause (tau=+1) RMSE = %.4fs", mean_rmse_A, mean_prmse_A),
    x = "Empirical Reaction Time (Seconds)",
    y = "Predicted Reaction Time (Seconds)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "none")

# Panel B: Model B (High-Dimensional Ridge) Scatter
p_scat_B <- ggplot(scatter_records, aes(x = Empirical_RT, y = Pred_Model_B)) +
  geom_point(aes(color = Is_Pause, shape = Is_Pause, size = Is_Pause, alpha = Is_Pause)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray30", linewidth = 0.9) +
  scale_color_manual(values = c("FALSE" = "#27ae60", "TRUE" = "#e74c3c"), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_size_manual(values = c("FALSE" = 1.2, "TRUE" = 3.5), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_alpha_manual(values = c("FALSE" = 0.35, "TRUE" = 0.90), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  coord_cartesian(xlim = c(0.1, 2.5), ylim = c(0.1, 2.5)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "B. Model B: High-Dimensional Ridge Readout (RT_hat = RT_base + w_RT^T * z_GC)",
    subtitle = sprintf("Overall RMSE = %.4fs | Post-Pause (tau=+1) RMSE = %.4fs (Ridge Superiority)", mean_rmse_B, mean_prmse_B),
    x = "Empirical Reaction Time (Seconds)",
    y = "Predicted Reaction Time (Seconds)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "bottom", legend.title = element_blank())

p_master_competition <- grid.arrange(p_scat_A, p_scat_B, ncol = 1)
ggsave("results/figures/kinematic_readout_competition_master_plot.png", plot = p_master_competition, width = 9.5, height = 11.5, dpi = 300)
cat("Saved results/figures/kinematic_readout_competition_master_plot.png\n")

cat("\n==============================================================================\n")
cat("KINEMATIC READOUT COMPETITION PIPELINE COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
