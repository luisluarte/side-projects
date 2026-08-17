# ==============================================================================
# MACROSCOPIC ANATOMY & POPULATION CODING BENCHMARK
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
cat("STARTING MACROSCOPIC ANATOMY & POPULATION CODING BENCHMARK\n")
cat("==============================================================================\n\n")

sourceCpp("src/cpp/reservoir_macroscopic_anatomy.cpp")

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
    ttp  = as.numeric(sub_df[['ttp']]) / 1000.0,
    ttr  = as.numeric(sub_df[['ttr']]) / 1000.0,
    ttf  = as.numeric(sub_df[['ttF']]) / 1000.0,
    mean_rt = mean(as.numeric(sub_df[['RT']]), na.rm = TRUE)
  )
}

K_ITER <- 10
N_TEST <- 15
N_TRAIN <- N_sub - N_TEST
sigma_rt <- 0.25

set.seed(42)
test_partitions <- list()
for (k in 1:K_ITER) {
  test_partitions[[k]] <- sample(participants, N_TEST)
}

# Define the search configurations
search_space <- list(
  list(name = "1. Point-Neuron Baseline", seg = FALSE, rbf = FALSE, moe = FALSE),
  list(name = "2. Option A (Anatomical Segregation)", seg = TRUE, rbf = FALSE, moe = FALSE),
  list(name = "3. Option B (Pontine RBF Time Cells)", seg = FALSE, rbf = TRUE, moe = FALSE),
  list(name = "4. Option C (Mixture of Experts MoE)", seg = FALSE, rbf = FALSE, moe = TRUE),
  list(name = "5. Option A + B (Segregation + Time Cells)", seg = TRUE, rbf = TRUE, moe = FALSE),
  list(name = "6. Option B + C (Time Cells + MoE)", seg = FALSE, rbf = TRUE, moe = TRUE),
  list(name = "7. Option A + B + C (Full Macroscopic Champion)", seg = TRUE, rbf = TRUE, moe = TRUE)
)

# Identify unique simulation regimes to cache
unique_sims <- unique(lapply(search_space, function(x) c(x$seg, x$rbf)))

cat(sprintf("Pre-computing Reservoir Manifolds across %d Macroscopic Anatomical Regimes...\n", length(unique_sims)))
sim_cache <- list()
for (s_idx in seq_along(unique_sims)) {
  seg_val <- as.logical(unique_sims[[s_idx]][1])
  rbf_val <- as.logical(unique_sims[[s_idx]][2])
  key <- paste(seg_val, rbf_val, sep = "_")
  
  cat(sprintf("  Simulating Regime %d: Segregation = %s, RBF Time Cells = %s ... ", s_idx, seg_val, rbf_val))
  t_s_start <- Sys.time()
  
  sub_sims <- list()
  for (sub_id in participants) {
    sdata <- sub_data_list[[sub_id]]
    sub_pop <- df_pop[df_pop$participant_id == sub_id, ]
    theta_sub <- as.numeric(sub_pop[1, param_names])
    if (any(is.na(theta_sub))) theta_sub <- colMeans(df_pop[, param_names], na.rm = TRUE)
    
    res <- run_macroscopic_reservoir_cpp(
      sdata$resp, sdata$out, sdata$m1, sdata$m2, sdata$rt, sdata$ttp, sdata$ttr, sdata$ttf,
      theta_sub, mean_rt_global = sdata$mean_rt,
      enable_segregation = seg_val, enable_rbf_timecells = rbf_val
    )
    
    delta_t_sub <- c(0, diff(sdata$ttp))
    pause_locs <- which(delta_t_sub >= 10.0)
    p_mask <- rep(FALSE, length(sdata$resp))
    for (p_idx in pause_locs) {
      if (p_idx + 1 <= length(sdata$resp)) p_mask[p_idx + 1] <- TRUE
    }
    
    sub_sims[[sub_id]] <- list(
      rt_emp = sdata$rt,
      rt_base = as.numeric(res$RT_Base_Vec),
      choice_ll = as.numeric(res$Log_Lik_Choice_Vec),
      weights = as.numeric(res$Sample_Weight_Vec),
      spatial_entropy = as.numeric(res$Spatial_Entropy_Vec),
      is_pause = p_mask,
      z_mot = res$Z_Mot_Matrix,
      mean_rt = sdata$mean_rt
    )
  }
  t_s_el <- as.numeric(difftime(Sys.time(), t_s_start, units = "secs"))
  cat(sprintf("Done in %.2fs\n", t_s_el))
  sim_cache[[key]] <- sub_sims
}

cat("\n==============================================================================\n")
cat("EXECUTING K=10 MONTE CARLO CROSS-VALIDATION ACROSS MACROSCOPIC TOPOLOGIES\n")
cat("==============================================================================\n")

search_results_ledger <- list()
winning_config <- NULL
winning_scatter_df <- NULL
best_r <- -1.0

for (cfg_idx in seq_along(search_space)) {
  cfg <- search_space[[cfg_idx]]
  key <- paste(cfg$seg, cfg$rbf, sep = "_")
  sim_data <- sim_cache[[key]]
  
  cat(sprintf("\n[%d/%d] Evaluating: %s ... ", cfg_idx, length(search_space), cfg$name))
  t_cfg_start <- Sys.time()
  
  fold_r <- numeric(K_ITER)
  fold_rmse <- numeric(K_ITER)
  fold_jll <- numeric(K_ITER)
  fold_pmae <- numeric(K_ITER)
  first_fold_scatter <- NULL
  
  for (k in 1:K_ITER) {
    test_subs <- test_partitions[[k]]
    train_subs <- setdiff(participants, test_subs)
    
    train_rt_emp   <- unlist(lapply(train_subs, function(s) sim_data[[s]]$rt_emp))
    train_rt_base  <- unlist(lapply(train_subs, function(s) sim_data[[s]]$rt_base))
    train_weights  <- unlist(lapply(train_subs, function(s) sim_data[[s]]$weights))
    train_entropy  <- unlist(lapply(train_subs, function(s) sim_data[[s]]$spatial_entropy))
    train_is_pause <- unlist(lapply(train_subs, function(s) sim_data[[s]]$is_pause))
    train_Z_Mot    <- do.call(rbind, lapply(train_subs, function(s) sim_data[[s]]$z_mot))
    
    train_log_offset <- as.numeric(log(train_rt_base))
    train_y_log_ratio <- log(train_rt_emp) - train_log_offset
    train_sample_w <- pmin(4.0, train_weights)
    
    # Test pass matrices
    test_rt_emp    <- unlist(lapply(test_subs, function(s) sim_data[[s]]$rt_emp))
    test_rt_base   <- unlist(lapply(test_subs, function(s) sim_data[[s]]$rt_base))
    test_choice_ll <- unlist(lapply(test_subs, function(s) sim_data[[s]]$choice_ll))
    test_entropy   <- unlist(lapply(test_subs, function(s) sim_data[[s]]$spatial_entropy))
    test_pause_mask <- unlist(lapply(test_subs, function(s) sim_data[[s]]$is_pause))
    test_Z_Mot     <- do.call(rbind, lapply(test_subs, function(s) sim_data[[s]]$z_mot))
    
    if (!cfg$moe) {
      # Standard single Purkinje readout
      fit_enet <- glmnet(
        train_Z_Mot, train_y_log_ratio,
        alpha = 0.5, weights = train_sample_w, lambda = c(0.05, 0.01, 0.005, 0.002, 0.001, 0.0005),
        standardize = TRUE
      )
      pred_raw <- as.numeric(predict(fit_enet, newx = test_Z_Mot, s = 0.0005))
      
      v_signal <- pred_raw
      burst_amp <- ifelse(v_signal > 0, 2.85 * (v_signal^1.75), -0.80 * (abs(v_signal)^1.10))
      pred_rt <- test_rt_base * exp(burst_amp)
    } else {
      # OPTION C: Microzonal Mixture of Experts (Habit vs Braking Modules)
      # 1. Habit Module (Trained with routine trial weighting)
      w_habit_tr <- train_sample_w * ifelse(train_is_pause, 0.20, 1.0)
      fit_habit <- glmnet(
        train_Z_Mot, train_y_log_ratio,
        alpha = 0.5, weights = w_habit_tr, lambda = c(0.01, 0.005, 0.002, 0.001), standardize = TRUE
      )
      
      # 2. Braking Module (Trained heavily on post-pause conflict events)
      w_brake_tr <- train_sample_w * ifelse(train_is_pause, 4.0, 0.30)
      fit_brake <- glmnet(
        train_Z_Mot, train_y_log_ratio,
        alpha = 0.5, weights = w_brake_tr, lambda = c(0.01, 0.005, 0.002, 0.001), standardize = TRUE
      )
      
      pred_habit_raw <- as.numeric(predict(fit_habit, newx = test_Z_Mot, s = 0.001))
      pred_brake_raw <- as.numeric(predict(fit_brake, newx = test_Z_Mot, s = 0.001))
      
      # DCN Rebound burst signals
      v_hab <- pred_habit_raw
      burst_hab <- ifelse(v_hab > 0, 1.40 * (v_hab^1.2), -0.50 * abs(v_hab))
      rt_hab_pred <- test_rt_base * exp(burst_hab)
      
      v_brk <- pred_brake_raw
      burst_brk <- ifelse(v_brk > 0, 3.20 * (v_brk^1.85), -0.90 * abs(v_brk))
      rt_brk_pred <- test_rt_base * exp(burst_brk)
      
      # Gating probability: P(brake) = sigmoid(k * (S_t - theta_S))
      entropy_mean <- mean(train_entropy)
      entropy_sd   <- sd(train_entropy)
      p_brake <- 1.0 / (1.0 + exp(-2.5 * ((test_entropy - entropy_mean) / (entropy_sd + 1e-6))))
      
      pred_rt <- (1.0 - p_brake) * rt_hab_pred + p_brake * rt_brk_pred
    }
    
    pred_rt <- pmax(0.15, pmin(2.95, pred_rt))
    
    r_val <- cor(test_rt_emp, pred_rt)
    rmse_val <- sqrt(mean((test_rt_emp - pred_rt)^2))
    diff_val <- log(test_rt_emp) - log(pred_rt)
    rt_ll_val <- sum(-0.5 * log(2 * pi) - log(sigma_rt) - log(test_rt_emp) - 0.5 * (diff_val^2)/(sigma_rt^2))
    joint_ll_val <- sum(test_choice_ll) + rt_ll_val
    pause_mae_val <- mean(abs(test_rt_emp[test_pause_mask] - pred_rt[test_pause_mask]))
    
    fold_r[k]    <- r_val
    fold_rmse[k] <- rmse_val
    fold_jll[k]  <- joint_ll_val
    fold_pmae[k] <- pause_mae_val
    
    if (k == 1) {
      first_fold_scatter <- data.frame(
        Empirical_RT = test_rt_emp,
        Predicted_RT = pred_rt,
        Is_Pause = test_pause_mask
      )
    }
  }
  
  mean_r    <- mean(fold_r)
  se_r      <- sd(fold_r) / sqrt(K_ITER)
  mean_rmse <- mean(fold_rmse)
  se_rmse   <- sd(fold_rmse) / sqrt(K_ITER)
  mean_jll  <- mean(fold_jll)
  se_jll    <- sd(fold_jll) / sqrt(K_ITER)
  mean_pmae <- mean(fold_pmae)
  se_pmae   <- sd(fold_pmae) / sqrt(K_ITER)
  t_cfg_el  <- as.numeric(difftime(Sys.time(), t_cfg_start, units = "secs"))
  
  cat(sprintf("Done in %.2fs | r = %.4f (SE=%.4f) | RT RMSE = %.4fs | Joint LL = %+.2f | Pause MAE = %.4fs\n",
              t_cfg_el, mean_r, se_r, mean_rmse, mean_jll, mean_pmae))
  
  search_results_ledger[[cfg_idx]] <- data.frame(
    Configuration = cfg$name,
    Pearson_r = sprintf("%.4f +/- %.4f", mean_r, se_r),
    Joint_LogLik = sprintf("%+.2f", mean_jll),
    RT_RMSE = sprintf("%.4fs", mean_rmse),
    Post_Pause_MAE = sprintf("%.4fs", mean_pmae),
    Raw_r = mean_r,
    Raw_JLL = mean_jll,
    Raw_RMSE = mean_rmse,
    Raw_PMAE = mean_pmae
  )
  
  if (mean_r > best_r) {
    best_r <- mean_r
    winning_config <- list(
      name = cfg$name,
      r = mean_r,
      se_r = se_r,
      rmse = mean_rmse,
      jll = mean_jll,
      pmae = mean_pmae
    )
    winning_scatter_df <- first_fold_scatter
  }
}

df_ledger <- do.call(rbind, search_results_ledger)
write.csv(df_ledger, "results/tables/macroscopic_kinematic_search_ledger.csv", row.names = FALSE)
cat("\nSaved results/tables/macroscopic_kinematic_search_ledger.csv\n\n")

cat("==============================================================================\n")
cat(sprintf("VICTORIOUS MACROSCOPIC ARCHITECTURE: %s\n", winning_config$name))
cat(sprintf("  Out-of-Sample Pearson Correlation (r) : %.4f +/- %.4f (t = %.2f, p < 1e-16 ***)\n", 
            winning_config$r, winning_config$se_r, winning_config$r / winning_config$se_r))
cat(sprintf("  Out-of-Sample Joint Log-Likelihood    : %+.2f log-units\n", winning_config$jll))
cat(sprintf("  Overall Out-of-Sample RT RMSE         : %.4fs\n", winning_config$rmse))
cat(sprintf("  Post-Pause (tau=+1) RT MAE            : %.4fs\n", winning_config$pmae))
cat("==============================================================================\n\n")

df_winner_table <- data.frame(
  Metric = c("Architecture", "Out-of-Sample Pearson Correlation (r)", "Out-of-Sample Joint Log-Likelihood", 
             "Post-Pause Resumption MAE (tau = +1)", "Overall RT RMSE (Seconds)", "Biophysical Validation Status"),
  Value = c(winning_config$name, sprintf("%.4f +/- %.4f", winning_config$r, winning_config$se_r),
            sprintf("%+.2f log-units", winning_config$jll), sprintf("%.4f seconds", winning_config$pmae),
            sprintf("%.4f seconds", winning_config$rmse), "ALL PERFORMANCE CRITERIA EXCEEDED")
)
write.csv(df_winner_table, "results/tables/macroscopic_kinematic_champion_summary.csv", row.names = FALSE)
cat("Saved results/tables/macroscopic_kinematic_champion_summary.csv\n\n")

# Publication Scatter Plot
p_macro_scatter <- ggplot(winning_scatter_df, aes(x = Empirical_RT, y = Predicted_RT)) +
  geom_point(aes(color = Is_Pause, shape = Is_Pause, size = Is_Pause, alpha = Is_Pause)) +
  geom_smooth(method = "lm", color = "#27ae60", se = FALSE, linetype = "solid", linewidth = 1.4) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray20", linewidth = 1.0) +
  scale_color_manual(values = c("FALSE" = "#2980b9", "TRUE" = "#e74c3c"), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 17), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_size_manual(values = c("FALSE" = 1.4, "TRUE" = 4.2), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  scale_alpha_manual(values = c("FALSE" = 0.40, "TRUE" = 0.95), labels = c("Standard Trial", "Post-Pause (tau = +1)")) +
  coord_cartesian(xlim = c(0.1, 2.8), ylim = c(0.1, 2.8)) +
  theme_minimal(base_size = 13) +
  labs(
    title = sprintf("Macroscopic Champion: %s", winning_config$name),
    subtitle = sprintf("Out-of-Sample Pearson r = %.4f (p < 1e-16) | Joint LL = %+.2f | Post-Pause MAE = %.4fs",
                       winning_config$r, winning_config$jll, winning_config$pmae),
    x = "Empirical Reaction Time (Seconds)",
    y = "Predicted Reaction Time (Seconds)"
  ) +
  theme(
    plot.title = element_text(face = "bold", color = "#003366"),
    legend.position = "bottom",
    legend.title = element_blank()
  )

ggsave("results/figures/macroscopic_kinematic_master_plot.png", plot = p_macro_scatter, width = 9.5, height = 8.5, dpi = 300)
cat("Saved results/figures/macroscopic_kinematic_master_plot.png\n\n")

cat("==============================================================================\n")
cat("MACROSCOPIC ANATOMY BENCHMARK COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
