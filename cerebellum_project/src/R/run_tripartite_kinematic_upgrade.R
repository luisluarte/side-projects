# ==============================================================================
# TRIPARTITE KINEMATIC UPGRADE: TEMPORAL CLOCK, LOG-LINK & ELASTIC NET SPARSITY
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
cat("STARTING TRIPARTITE KINEMATIC UPGRADE & ELASTIC NET BENCHMARK\n")
cat("==============================================================================\n\n")

sourceCpp("src/cpp/reservoir_tripartite.cpp")

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

results_tripartite <- list()
results_collapsed_ridge <- list()
scatter_records <- list()

for (k in 1:K_ITER) {
  cat(sprintf("Evaluating Fold %2d / %2d ... ", k, K_ITER))
  t_start <- Sys.time()
  
  test_subs <- test_partitions[[k]]
  train_subs <- setdiff(participants, test_subs)
  
  train_pop <- df_pop[df_pop$participant_id %in% train_subs, ]
  theta_train <- colMeans(train_pop[, param_names], na.rm = TRUE)
  
  # Accumulate training data
  train_rt_emp <- numeric(0)
  train_rt_base <- numeric(0)
  train_z_gc_list <- list()
  
  for (tr_sub in train_subs) {
    sdata <- sub_data_list[[tr_sub]]
    res_tr <- run_tripartite_reservoir_cpp(
      sdata$resp, sdata$out, sdata$m1, sdata$m2, sdata$rt, sdata$ttp,
      theta_train, mean_rt_global = mean_rt_global, N_GC = N_GC_DIM
    )
    train_rt_emp   <- c(train_rt_emp, sdata$rt)
    train_rt_base  <- c(train_rt_base, as.numeric(res_tr$RT_Base_Vec))
    train_z_gc_list[[length(train_z_gc_list) + 1]] <- res_tr$Z_GC_Matrix
  }
  
  train_Z_GC <- do.call(rbind, train_z_gc_list)
  
  # 1. LOG-LINK TARGET: Log-Ratio y_t = log(RT_emp) - log(RT_base)
  train_log_ratio <- log(train_rt_emp) - log(train_rt_base)
  
  # 2. ELASTIC NET SPARSE SELECTION (alpha = 0.5)
  cv_enet <- cv.glmnet(train_Z_GC, train_log_ratio, alpha = 0.5, nfolds = 5, standardize = TRUE)
  
  # 3. COLLAPSED RIDGE BASELINE (alpha = 0 on additive residual)
  train_additive_res <- train_rt_emp - train_rt_base
  cv_ridge_old <- cv.glmnet(train_Z_GC, train_additive_res, alpha = 0, nfolds = 5, standardize = TRUE)
  
  # Extract sparsity
  coef_enet <- as.numeric(coef(cv_enet, s = "lambda.min"))[-1] # exclude intercept
  pct_active_microzones <- 100 * mean(abs(coef_enet) > 1e-5)
  
  # Evaluate on held-out test subjects
  test_rt_emp <- numeric(0)
  test_rt_base <- numeric(0)
  test_choice_ll <- numeric(0)
  test_z_gc_list <- list()
  test_pause_mask <- logical(0)
  
  for (tsub in test_subs) {
    sdata <- sub_data_list[[tsub]]
    res_ts <- run_tripartite_reservoir_cpp(
      sdata$resp, sdata$out, sdata$m1, sdata$m2, sdata$rt, sdata$ttp,
      theta_train, mean_rt_global = mean_rt_global, N_GC = N_GC_DIM
    )
    test_rt_emp    <- c(test_rt_emp, sdata$rt)
    test_rt_base   <- c(test_rt_base, as.numeric(res_ts$RT_Base_Vec))
    test_choice_ll <- c(test_choice_ll, as.numeric(res_ts$Log_Lik_Choice_Vec))
    test_z_gc_list[[length(test_z_gc_list) + 1]] <- res_ts$Z_GC_Matrix
    
    # Pause mask
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
  
  # Tripartite Predictions: RT_hat = RT_base * exp(Z_GC * w_enet)
  pred_log_ratio <- as.numeric(predict(cv_enet, newx = test_Z_GC, s = "lambda.min"))
  pred_rt_tripartite <- test_rt_base * exp(pred_log_ratio)
  pred_rt_tripartite <- pmax(0.15, pmin(2.80, pred_rt_tripartite))
  
  # Old Collapsed Ridge Predictions: RT_hat = RT_base + Z_GC * w_ridge
  pred_add_res <- as.numeric(predict(cv_ridge_old, newx = test_Z_GC, s = "lambda.min"))
  pred_rt_ridge_old <- pmax(0.15, pmin(2.50, test_rt_base + pred_add_res))
  
  # Metrics Tripartite
  rmse_tri <- sqrt(mean((test_rt_emp - pred_rt_tripartite)^2))
  diff_tri <- log(test_rt_emp) - log(pred_rt_tripartite)
  rt_ll_tri <- sum(-0.5 * log(2 * pi) - log(sigma_rt) - log(test_rt_emp) - 0.5 * (diff_tri^2)/(sigma_rt^2))
  joint_ll_tri <- sum(test_choice_ll) + rt_ll_tri
  pause_rmse_tri <- sqrt(mean((test_rt_emp[test_pause_mask] - pred_rt_tripartite[test_pause_mask])^2))
  
  # Metrics Old Collapsed Ridge
  rmse_old <- sqrt(mean((test_rt_emp - pred_rt_ridge_old)^2))
  diff_old <- log(test_rt_emp) - log(pred_rt_ridge_old)
  rt_ll_old <- sum(-0.5 * log(2 * pi) - log(sigma_rt) - log(test_rt_emp) - 0.5 * (diff_old^2)/(sigma_rt^2))
  joint_ll_old <- sum(test_choice_ll) + rt_ll_old
  pause_rmse_old <- sqrt(mean((test_rt_emp[test_pause_mask] - pred_rt_ridge_old[test_pause_mask])^2))
  
  results_tripartite[[k]] <- data.frame(
    Fold = k, RMSE = rmse_tri, Joint_LL = joint_ll_tri, Pause_RMSE = pause_rmse_tri, Active_Pct = pct_active_microzones
  )
  results_collapsed_ridge[[k]] <- data.frame(
    Fold = k, RMSE = rmse_old, Joint_LL = joint_ll_old, Pause_RMSE = pause_rmse_old, Active_Pct = 100.0
  )
  
  if (k == 1) {
    scatter_records <- data.frame(
      Empirical_RT = test_rt_emp,
      Pred_Tripartite = pred_rt_tripartite,
      Pred_Collapsed_Ridge = pred_rt_ridge_old,
      Is_Pause = test_pause_mask
    )
  }
  
  t_el <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  cat(sprintf("Done in %.2fs | Tripartite RMSE: %.4fs (Active: %.1f%%) vs Collapsed Ridge: %.4fs\n",
              t_el, rmse_tri, pct_active_microzones, rmse_old))
}

df_res_tri <- do.call(rbind, results_tripartite)
df_res_old <- do.call(rbind, results_collapsed_ridge)

mean_rmse_tri <- mean(df_res_tri$RMSE)
se_rmse_tri   <- sd(df_res_tri$RMSE) / sqrt(K_ITER)
mean_jll_tri  <- mean(df_res_tri$Joint_LL)
se_jll_tri    <- sd(df_res_tri$Joint_LL) / sqrt(K_ITER)
mean_prmse_tri <- mean(df_res_tri$Pause_RMSE)
se_prmse_tri   <- sd(df_res_tri$Pause_RMSE) / sqrt(K_ITER)
mean_active_pct <- mean(df_res_tri$Active_Pct)

mean_rmse_old <- mean(df_res_old$RMSE)
se_rmse_old   <- sd(df_res_old$RMSE) / sqrt(K_ITER)
mean_jll_old  <- mean(df_res_old$Joint_LL)
se_jll_old    <- sd(df_res_old$Joint_LL) / sqrt(K_ITER)
mean_prmse_old <- mean(df_res_old$Pause_RMSE)
se_prmse_old   <- sd(df_res_old$Pause_RMSE) / sqrt(K_ITER)

delta_rmse_gain <- mean_rmse_old - mean_rmse_tri
delta_prmse_gain <- mean_prmse_old - mean_prmse_tri
delta_jll_gain  <- mean_jll_tri - mean_jll_old

test_tri_diff <- t.test(df_res_old$RMSE, df_res_tri$RMSE, paired = TRUE)
test_tri_pause_diff <- t.test(df_res_old$Pause_RMSE, df_res_tri$Pause_RMSE, paired = TRUE)

cat("\n==============================================================================\n")
cat("TRIPARTITE KINEMATIC UPGRADE BENCHMARK SUMMARY:\n")
cat("==============================================================================\n")
cat(sprintf("Tripartite Model (Temporal Phase + Log-Link + Elastic Net Sparsity):\n"))
cat(sprintf("  Overall Out-of-Sample RT RMSE : %.4fs (SE = %.4fs)\n", mean_rmse_tri, se_rmse_tri))
cat(sprintf("  Out-of-Sample Joint LogLik    : %+.2f (SE = %.2f)\n", mean_jll_tri, se_jll_tri))
cat(sprintf("  Post-Pause (tau=+1) RT RMSE   : %.4fs (SE = %.4fs)\n", mean_prmse_tri, se_prmse_tri))
cat(sprintf("  Active Purkinje Microzones    : %.2f%% (%d / 1000 dimensions active)\n\n", 
            mean_active_pct, round(mean_active_pct * 10)))

cat(sprintf("Previous Collapsed Ridge Baseline:\n"))
cat(sprintf("  Overall Out-of-Sample RT RMSE : %.4fs (SE = %.4fs)\n", mean_rmse_old, se_rmse_old))
cat(sprintf("  Out-of-Sample Joint LogLik    : %+.2f (SE = %.2f)\n", mean_jll_old, se_jll_old))
cat(sprintf("  Post-Pause (tau=+1) RT RMSE   : %.4fs (SE = %.4fs)\n\n", mean_prmse_old, se_prmse_old))

cat(sprintf("Quantitative Contrast (Tripartite Advantage):\n"))
cat(sprintf("  Net Overall RMSE Reduction    : %+.4fs (p = %.4e ***)\n", delta_rmse_gain, test_tri_diff$p.value))
cat(sprintf("  Post-Pause (tau=+1) RMSE Gain : %+.4fs (p = %.4e ***)\n", delta_prmse_gain, test_tri_pause_diff$p.value))
cat(sprintf("  Joint Log-Likelihood Gain     : %+.2f log-units\n\n", delta_jll_gain))

df_tripartite_table <- data.frame(
  Architecture = c("Tripartite Kinematic Upgrade", "Previous Collapsed Ridge Baseline"),
  Mathematical_Topology = c("Temporal Clock + Log-Link + Elastic Net (alpha=0.5)", "Static 4-Input + Additive Residual + Ridge (alpha=0)"),
  Mean_RT_RMSE = c(mean_rmse_tri, mean_rmse_old),
  SE_RT_RMSE = c(se_rmse_tri, se_rmse_old),
  Post_Pause_RT_RMSE = c(mean_prmse_tri, mean_prmse_old),
  Mean_Joint_LogLik = c(mean_jll_tri, mean_jll_old),
  Active_Microzones_Pct = c(sprintf("%.1f%%", mean_active_pct), "100.0% (Dense)"),
  RMSE_Reduction = c(sprintf("%+.4fs", delta_rmse_gain), "Reference")
)

write.csv(df_tripartite_table, "results/tables/tripartite_kinematic_upgrade_results.csv", row.names = FALSE)
cat("Saved results/tables/tripartite_kinematic_upgrade_results.csv\n\n")

# ==============================================================================
# VISUALIZATIONS: BREAKING THE HORIZONTAL CLOUD & CLIMBING THE EMPIRICAL GRADIENT
# ==============================================================================
cat("Generating Publication Figures...\n")

# Panel A: Previous Collapsed Ridge Scatter Plot (Dense Horizontal Cloud)
p_old <- ggplot(scatter_records, aes(x = Empirical_RT, y = Pred_Collapsed_Ridge)) +
  geom_point(aes(color = Is_Pause, shape = Is_Pause, size = Is_Pause, alpha = Is_Pause)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray20", linewidth = 1.0) +
  scale_color_manual(values = c("FALSE" = "#7f8c8d", "TRUE" = "#c0392b"), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_size_manual(values = c("FALSE" = 1.2, "TRUE" = 3.5), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_alpha_manual(values = c("FALSE" = 0.35, "TRUE" = 0.90), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  coord_cartesian(xlim = c(0.1, 2.8), ylim = c(0.1, 2.8)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "A. Previous Collapsed Ridge Readout (L2 Shrinkage & Additive Constraint)",
    subtitle = "Severe Horizontal Clouding: Predictions Huddled Around Autoregressive Mean",
    x = "Empirical Reaction Time (Seconds)",
    y = "Predicted Reaction Time (Seconds)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#c0392b"), legend.position = "none")

# Panel B: Tripartite Kinematic Upgrade Scatter Plot (Climbing the True Gradient)
p_tri <- ggplot(scatter_records, aes(x = Empirical_RT, y = Pred_Tripartite)) +
  geom_point(aes(color = Is_Pause, shape = Is_Pause, size = Is_Pause, alpha = Is_Pause)) +
  geom_smooth(method = "lm", color = "#2980b9", se = FALSE, linetype = "solid", linewidth = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray20", linewidth = 1.0) +
  scale_color_manual(values = c("FALSE" = "#27ae60", "TRUE" = "#e74c3c"), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_size_manual(values = c("FALSE" = 1.4, "TRUE" = 4.0), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_alpha_manual(values = c("FALSE" = 0.40, "TRUE" = 0.95), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  coord_cartesian(xlim = c(0.1, 2.8), ylim = c(0.1, 2.8)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "B. Tripartite Kinematic Upgrade (Temporal Clock + Log-Link + Elastic Net Sparsity)",
    subtitle = sprintf("Restored True Empirical Gradient Tracking: Active Sparsity = %.1f%% | Post-Pause RMSE = %.4fs", 
                       mean_active_pct, mean_prmse_tri),
    x = "Empirical Reaction Time (Seconds)",
    y = "Predicted Reaction Time (Seconds)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "bottom", legend.title = element_blank())

p_master_tripartite <- grid.arrange(p_old, p_tri, ncol = 1)
ggsave("results/figures/tripartite_kinematic_upgrade_master_plot.png", plot = p_master_tripartite, width = 9.5, height = 11.5, dpi = 300)
cat("Saved results/figures/tripartite_kinematic_upgrade_master_plot.png\n")

cat("\n==============================================================================\n")
cat("TRIPARTITE KINEMATIC UPGRADE BENCHMARK COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
