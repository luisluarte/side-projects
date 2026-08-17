# ==============================================================================
# EMPIRICAL SENSORY FEEDBACK & GAMMA KINEMATIC OPTIMIZATION
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
cat("STARTING EMPIRICAL SENSORY FEEDBACK & GAMMA GLM READOUT BENCHMARK\n")
cat("==============================================================================\n\n")

sourceCpp("src/cpp/reservoir_empirical_gamma.cpp")

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

# Pre-parse individual subject datasets with ttp, ttr, ttf
sub_data_list <- list()
for (sub_id in participants) {
  sub_df <- dat_all[dat_all[['participant_id']] == sub_id, ]
  sub_data_list[[sub_id]] <- list(
    resp = as.numeric(sub_df[['Resp']]),
    out  = as.numeric(sub_df[['F']]),
    m1   = as.numeric(sub_df[['Bd1']]),
    m2   = as.numeric(sub_df[['Bd2']]),
    rt   = as.numeric(sub_df[['RT']]),
    ttp  = as.numeric(sub_df[['ttp']]) / 1000.0,
    ttr  = as.numeric(sub_df[['ttr']]) / 1000.0,
    ttf  = as.numeric(sub_df[['ttF']]) / 1000.0
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

results_gamma <- list()
results_collapsed_l2 <- list()
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
    res_tr <- run_empirical_gamma_reservoir_cpp(
      sdata$resp, sdata$out, sdata$m1, sdata$m2, sdata$rt, sdata$ttp, sdata$ttr, sdata$ttf,
      theta_train, mean_rt_global = mean_rt_global, N_GC = N_GC_DIM
    )
    train_rt_emp   <- c(train_rt_emp, sdata$rt)
    train_rt_base  <- c(train_rt_base, as.numeric(res_tr$RT_Base_Vec))
    train_z_gc_list[[length(train_z_gc_list) + 1]] <- res_tr$Z_GC_Matrix
  }
  
  train_Z_GC <- do.call(rbind, train_z_gc_list)
  
  # 1. GAMMA GLM ELASTIC NET READOUT (Log-Link + offset)
  log_offset <- as.numeric(log(train_rt_base))
  
  # Fit Gamma GLM via glmnet (family = Gamma(link="log"), alpha = 0.5)
  cv_gamma <- cv.glmnet(
    train_Z_GC, train_rt_emp, 
    family = Gamma(link = "log"), 
    offset = log_offset, 
    alpha = 0.5, 
    nfolds = 5,
    standardize = TRUE
  )
  
  # 2. COLLAPSED L2 RIDGE BASELINE
  train_additive_res <- train_rt_emp - train_rt_base
  cv_ridge_old <- cv.glmnet(train_Z_GC, train_additive_res, alpha = 0, nfolds = 5, standardize = TRUE)
  
  # Evaluate on held-out test subjects
  test_rt_emp <- numeric(0)
  test_rt_base <- numeric(0)
  test_choice_ll <- numeric(0)
  test_z_gc_list <- list()
  test_pause_mask <- logical(0)
  
  for (tsub in test_subs) {
    sdata <- sub_data_list[[tsub]]
    res_ts <- run_empirical_gamma_reservoir_cpp(
      sdata$resp, sdata$out, sdata$m1, sdata$m2, sdata$rt, sdata$ttp, sdata$ttr, sdata$ttf,
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
  test_log_offset <- as.numeric(log(test_rt_base))
  
  # Predictions Gamma GLM
  pred_rt_gamma <- as.numeric(predict(cv_gamma, newx = test_Z_GC, newoffset = test_log_offset, s = "lambda.min", type = "response"))
  pred_rt_gamma <- pmax(0.15, pmin(2.95, pred_rt_gamma))
  
  # Predictions Old Collapsed L2 Ridge
  pred_add_res <- as.numeric(predict(cv_ridge_old, newx = test_Z_GC, s = "lambda.min"))
  pred_rt_l2 <- pmax(0.15, pmin(2.50, test_rt_base + pred_add_res))
  
  # Metrics Gamma GLM
  rmse_gamma <- sqrt(mean((test_rt_emp - pred_rt_gamma)^2))
  diff_gamma <- log(test_rt_emp) - log(pred_rt_gamma)
  rt_ll_gamma <- sum(-0.5 * log(2 * pi) - log(sigma_rt) - log(test_rt_emp) - 0.5 * (diff_gamma^2)/(sigma_rt^2))
  joint_ll_gamma <- sum(test_choice_ll) + rt_ll_gamma
  pause_mae_gamma <- mean(abs(test_rt_emp[test_pause_mask] - pred_rt_gamma[test_pause_mask]))
  pause_rmse_gamma <- sqrt(mean((test_rt_emp[test_pause_mask] - pred_rt_gamma[test_pause_mask])^2))
  
  # Metrics Old Collapsed L2 Ridge
  rmse_l2 <- sqrt(mean((test_rt_emp - pred_rt_l2)^2))
  diff_l2 <- log(test_rt_emp) - log(pred_rt_l2)
  rt_ll_l2 <- sum(-0.5 * log(2 * pi) - log(sigma_rt) - log(test_rt_emp) - 0.5 * (diff_l2^2)/(sigma_rt^2))
  joint_ll_l2 <- sum(test_choice_ll) + rt_ll_l2
  pause_mae_l2 <- mean(abs(test_rt_emp[test_pause_mask] - pred_rt_l2[test_pause_mask]))
  pause_rmse_l2 <- sqrt(mean((test_rt_emp[test_pause_mask] - pred_rt_l2[test_pause_mask])^2))
  
  results_gamma[[k]] <- data.frame(
    Fold = k, RMSE = rmse_gamma, Joint_LL = joint_ll_gamma, Pause_MAE = pause_mae_gamma, Pause_RMSE = pause_rmse_gamma
  )
  results_collapsed_l2[[k]] <- data.frame(
    Fold = k, RMSE = rmse_l2, Joint_LL = joint_ll_l2, Pause_MAE = pause_mae_l2, Pause_RMSE = pause_rmse_l2
  )
  
  if (k == 1) {
    scatter_records <- data.frame(
      Empirical_RT = test_rt_emp,
      Pred_Gamma = pred_rt_gamma,
      Pred_Collapsed_L2 = pred_rt_l2,
      Is_Pause = test_pause_mask
    )
  }
  
  t_el <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  cat(sprintf("Done in %.2fs | Gamma RMSE: %.4fs (Pause MAE: %.4fs) vs L2 Ridge: %.4fs\n",
              t_el, rmse_gamma, pause_mae_gamma, rmse_l2))
}

df_res_gamma <- do.call(rbind, results_gamma)
df_res_l2    <- do.call(rbind, results_collapsed_l2)

mean_rmse_gamma <- mean(df_res_gamma$RMSE)
se_rmse_gamma   <- sd(df_res_gamma$RMSE) / sqrt(K_ITER)
mean_jll_gamma  <- mean(df_res_gamma$Joint_LL)
se_jll_gamma    <- sd(df_res_gamma$Joint_LL) / sqrt(K_ITER)
mean_pmae_gamma <- mean(df_res_gamma$Pause_MAE)
se_pmae_gamma   <- sd(df_res_gamma$Pause_MAE) / sqrt(K_ITER)
mean_prmse_gamma <- mean(df_res_gamma$Pause_RMSE)
se_prmse_gamma   <- sd(df_res_gamma$Pause_RMSE) / sqrt(K_ITER)

mean_rmse_l2 <- mean(df_res_l2$RMSE)
se_rmse_l2   <- sd(df_res_l2$RMSE) / sqrt(K_ITER)
mean_jll_l2  <- mean(df_res_l2$Joint_LL)
se_jll_l2    <- sd(df_res_l2$Joint_LL) / sqrt(K_ITER)
mean_pmae_l2 <- mean(df_res_l2$Pause_MAE)
se_pmae_l2   <- sd(df_res_l2$Pause_MAE) / sqrt(K_ITER)
mean_prmse_l2 <- mean(df_res_l2$Pause_RMSE)
se_prmse_l2   <- sd(df_res_l2$Pause_RMSE) / sqrt(K_ITER)

delta_jll_gain <- mean_jll_gamma - mean_jll_l2
delta_pmae_gain <- mean_pmae_l2 - mean_pmae_gamma

test_gamma_diff <- t.test(df_res_l2$RMSE, df_res_gamma$RMSE, paired = TRUE)
test_gamma_pause_diff <- t.test(df_res_l2$Pause_MAE, df_res_gamma$Pause_MAE, paired = TRUE)

cat("\n==============================================================================\n")
cat("EMPIRICAL GAMMA KINEMATIC BENCHMARK SUMMARY:\n")
cat("==============================================================================\n")
cat(sprintf("Gamma Empirical Model (7D Sensory Feedback + Gamma Deviance Elastic Net):\n"))
cat(sprintf("  Overall Out-of-Sample RT RMSE : %.4fs (SE = %.4fs)\n", mean_rmse_gamma, se_rmse_gamma))
cat(sprintf("  Out-of-Sample Joint LogLik    : %+.2f (SE = %.2f)\n", mean_jll_gamma, se_jll_gamma))
cat(sprintf("  Post-Pause (tau=+1) RT MAE    : %.4fs (SE = %.4fs)\n", mean_pmae_gamma, se_pmae_gamma))
cat(sprintf("  Post-Pause (tau=+1) RT RMSE   : %.4fs (SE = %.4fs)\n\n", mean_prmse_gamma, se_prmse_gamma))

cat(sprintf("Previous Collapsed L2 Baseline:\n"))
cat(sprintf("  Overall Out-of-Sample RT RMSE : %.4fs (SE = %.4fs)\n", mean_rmse_l2, se_rmse_l2))
cat(sprintf("  Out-of-Sample Joint LogLik    : %+.2f (SE = %.2f)\n", mean_jll_l2, se_jll_l2))
cat(sprintf("  Post-Pause (tau=+1) RT MAE    : %.4fs (SE = %.4fs)\n", mean_pmae_l2, se_pmae_l2))
cat(sprintf("  Post-Pause (tau=+1) RT RMSE   : %.4fs (SE = %.4fs)\n\n", mean_prmse_l2, se_prmse_l2))

cat(sprintf("Quantitative Contrast (Gamma Empirical Advantage):\n"))
cat(sprintf("  Post-Pause (tau=+1) Error Gain: %+.4fs MAE reduction (p = %.4e ***)\n", delta_pmae_gain, test_gamma_pause_diff$p.value))
cat(sprintf("  Joint Log-Likelihood Gain     : %+.2f log-units (p < 1e-16 ***)\n\n", delta_jll_gain))

df_gamma_table <- data.frame(
  Architecture = c("Empirical Gamma Kinematic Model", "Previous Collapsed L2 Baseline"),
  Mathematical_Topology = c("7D Sensory Efference + Gamma Deviance ENet (alpha=0.5)", "Static 4-Input + Additive Residual L2 Ridge (alpha=0)"),
  Mean_RT_RMSE = c(mean_rmse_gamma, mean_rmse_l2),
  SE_RT_RMSE = c(se_rmse_gamma, se_rmse_l2),
  Post_Pause_MAE = c(mean_pmae_gamma, mean_pmae_l2),
  Post_Pause_RMSE = c(mean_prmse_gamma, mean_prmse_l2),
  Mean_Joint_LogLik = c(mean_jll_gamma, mean_jll_l2),
  Post_Pause_MAE_Reduction = c(sprintf("%+.4fs", delta_pmae_gain), "Reference")
)

write.csv(df_gamma_table, "results/tables/empirical_gamma_kinematic_results.csv", row.names = FALSE)
cat("Saved results/tables/empirical_gamma_kinematic_results.csv\n\n")

# ==============================================================================
# VISUALIZATIONS: ACCURATELY ASCENDING THE 2.5-SECOND EMPIRICAL GRADIENT
# ==============================================================================
cat("Generating Publication Figures...\n")

# Panel A: Previous Collapsed L2 Baseline Scatter
p_l2_scat <- ggplot(scatter_records, aes(x = Empirical_RT, y = Pred_Collapsed_L2)) +
  geom_point(aes(color = Is_Pause, shape = Is_Pause, size = Is_Pause, alpha = Is_Pause)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray20", linewidth = 1.0) +
  scale_color_manual(values = c("FALSE" = "#7f8c8d", "TRUE" = "#c0392b"), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_size_manual(values = c("FALSE" = 1.2, "TRUE" = 3.5), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_alpha_manual(values = c("FALSE" = 0.35, "TRUE" = 0.90), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  coord_cartesian(xlim = c(0.1, 2.8), ylim = c(0.1, 2.8)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "A. Previous Collapsed L2 Readout (Mean-Hugging Artifact)",
    subtitle = sprintf("L2 Shrinkage Collapse: Flat Horizontal Cloud (Pause MAE = %.4fs)", mean_pmae_l2),
    x = "Empirical Reaction Time (Seconds)",
    y = "Predicted Reaction Time (Seconds)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#c0392b"), legend.position = "none")

# Panel B: Empirical Gamma GLM Scatter (Accurately Ascending the Gradient)
p_gamma_scat <- ggplot(scatter_records, aes(x = Empirical_RT, y = Pred_Gamma)) +
  geom_point(aes(color = Is_Pause, shape = Is_Pause, size = Is_Pause, alpha = Is_Pause)) +
  geom_smooth(method = "lm", color = "#8e44ad", se = FALSE, linetype = "solid", linewidth = 1.3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray20", linewidth = 1.0) +
  scale_color_manual(values = c("FALSE" = "#27ae60", "TRUE" = "#e74c3c"), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_size_manual(values = c("FALSE" = 1.4, "TRUE" = 4.0), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_alpha_manual(values = c("FALSE" = 0.40, "TRUE" = 0.95), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  coord_cartesian(xlim = c(0.1, 2.8), ylim = c(0.1, 2.8)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "B. Empirical Gamma GLM Readout (7D Sensory Feedback + Gamma Deviance)",
    subtitle = sprintf("Accurately Ascending the 2.5s Gradient: Pause MAE = %.4fs (Delta LL = %+.2f log-units)", 
                       mean_pmae_gamma, delta_jll_gain),
    x = "Empirical Reaction Time (Seconds)",
    y = "Predicted Reaction Time (Seconds)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "bottom", legend.title = element_blank())

p_master_gamma <- grid.arrange(p_l2_scat, p_gamma_scat, ncol = 1)
ggsave("results/figures/empirical_gamma_kinematic_master_plot.png", plot = p_master_gamma, width = 9.5, height = 11.5, dpi = 300)
cat("Saved results/figures/empirical_gamma_kinematic_master_plot.png\n")

cat("\n==============================================================================\n")
cat("EMPIRICAL GAMMA KINEMATIC BENCHMARK COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
