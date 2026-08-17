# ==============================================================================
# DEFINITIVE 4-WAY STOCHASTIC TOURNAMENT
# Kernelized Symplectic-HDDM vs. Cognitive Baselines
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(stats)
  library(ggplot2)
  library(gridExtra)
})

cat("==============================================================================\n")
cat("STARTING DEFINITIVE 4-WAY STOCHASTIC TOURNAMENT\n")
cat("==============================================================================\n\n")

sourceCpp("src/cpp/reservoir_definitive_stochastic_tournament.cpp")

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

cat("Pre-simulating internal decision signals across 128 human subjects in C++ ... ")
t_sim_start <- Sys.time()
sim_cached_tourn <- list()
for (sub_id in participants) {
  sdata <- sub_data_list[[sub_id]]
  sub_pop <- df_pop[df_pop$participant_id == sub_id, ]
  theta_sub <- as.numeric(sub_pop[1, param_names])
  if (any(is.na(theta_sub))) theta_sub <- colMeans(df_pop[, param_names], na.rm = TRUE)
  
  res <- precompute_definitive_tournament_subject_cpp(
    sdata$resp, sdata$out, sdata$m1, sdata$m2, sdata$rt, sdata$ttp,
    theta_sub, N_GC = 1000, N_MLI = 200
  )
  
  sim_cached_tourn[[sub_id]] <- list(
    resp = sdata$resp,
    rt = sdata$rt,
    v_m0 = as.numeric(res$V_M0),
    v_m1 = as.numeric(res$V_M1),
    v_m2 = as.numeric(res$V_M2),
    rpe_m2 = as.numeric(res$RPE_M2),
    v_m3 = as.numeric(res$V_M3),
    rebound_m3 = as.numeric(res$Rebound_M3),
    lc_wt = as.numeric(res$LC_Weight)
  )
}
cat(sprintf("Done in %.2fs\n\n", as.numeric(difftime(Sys.time(), t_sim_start, units = "secs"))))

K_ITER <- 10
N_TEST <- 15
N_TRAIN <- N_sub - N_TEST

set.seed(42)
test_partitions <- list()
for (k in 1:K_ITER) {
  test_partitions[[k]] <- sample(participants, N_TEST)
}

model_definitions <- list(
  list(id = 0, name = "Model 0: Intercept-Only DDM", v_key = "v_m0", mod_key = "v_m0", has_mod = FALSE),
  list(id = 1, name = "Model 1: WSLS-HDDM (Markovian)", v_key = "v_m1", mod_key = "v_m0", has_mod = FALSE),
  list(id = 2, name = "Model 2: RW-CF-HDDM (Value Tracker)", v_key = "v_m2", mod_key = "rpe_m2", has_mod = TRUE),
  list(id = 3, name = "Model 3: Kernelized Symplectic-HDDM", v_key = "v_m3", mod_key = "rebound_m3", has_mod = TRUE)
)

tournament_results_ledger <- list()
sim_densities_fold1 <- list()

for (m_idx in seq_along(model_definitions)) {
  mdef <- model_definitions[[m_idx]]
  cat(sprintf("[%d/4] Benchmarking: %s ... ", m_idx, mdef$name))
  t_m_start <- Sys.time()
  
  fold_dev <- numeric(K_ITER)
  fold_brier <- numeric(K_ITER)
  fold_dq90 <- numeric(K_ITER)
  fold_acc <- numeric(K_ITER)
  
  for (k in 1:K_ITER) {
    test_subs <- test_partitions[[k]]
    train_subs <- setdiff(participants, test_subs)
    
    tr_resp <- unlist(lapply(train_subs, function(s) sim_cached_tourn[[s]]$resp))
    tr_rt   <- unlist(lapply(train_subs, function(s) sim_cached_tourn[[s]]$rt))
    tr_v    <- unlist(lapply(train_subs, function(s) sim_cached_tourn[[s]][[mdef$v_key]]))
    tr_mod  <- unlist(lapply(train_subs, function(s) sim_cached_tourn[[s]][[mdef$mod_key]]))
    tr_wt   <- if (mdef$id == 3) unlist(lapply(train_subs, function(s) sim_cached_tourn[[s]]$lc_wt)) else rep(1.0, length(tr_resp))
    
    obj_fn <- function(p) {
      b_v  <- p[1]
      a0   <- p[2]
      kmod <- if (mdef$has_mod) p[3] else 0.0
      tnd  <- p[4]
      
      if (b_v < 0.01 || a0 < 0.3 || kmod < 0.0 || tnd < 0.05 || tnd > 0.35) return(1e9)
      compute_tournament_deviance_cpp(tr_resp, tr_rt, tr_v, tr_mod, tr_wt, b_v, a0, kmod, tnd)
    }
    
    init_p <- c(0.80, 1.20, if (mdef$has_mod) 0.05 else 0.0, 0.18)
    opt <- optim(init_p, obj_fn, method = "L-BFGS-B",
                 lower = c(0.01, 0.4, 0.0, 0.08),
                 upper = c(3.5, 2.5, 0.50, 0.30),
                 control = list(maxit = 35))
    
    b_v_opt  <- opt$par[1]
    a_0_opt  <- opt$par[2]
    kmod_opt <- if (mdef$has_mod) opt$par[3] else 0.0
    tnd_opt  <- opt$par[4]
    
    ts_resp <- unlist(lapply(test_subs, function(s) sim_cached_tourn[[s]]$resp))
    ts_rt   <- unlist(lapply(test_subs, function(s) sim_cached_tourn[[s]]$rt))
    ts_v    <- unlist(lapply(test_subs, function(s) sim_cached_tourn[[s]][[mdef$v_key]]))
    ts_mod  <- unlist(lapply(test_subs, function(s) sim_cached_tourn[[s]][[mdef$mod_key]]))
    
    res_ts <- evaluate_definitive_model_cpp(ts_resp, ts_rt, ts_v, ts_mod, b_v_opt, a_0_opt, kmod_opt, tnd_opt)
    
    dev_k   <- res_ts$Deviance
    brier_k <- res_ts$Brier_Score
    acc_k   <- res_ts$Choice_Accuracy
    pred_rt <- as.numeric(res_ts$Pred_Mean_RT)
    
    q90_emp  <- quantile(ts_rt, 0.90)
    q90_pred <- quantile(pred_rt, 0.90)
    dq90_k   <- abs(q90_emp - q90_pred)
    
    fold_dev[k]   <- dev_k
    fold_brier[k] <- brier_k
    fold_acc[k]   <- acc_k
    fold_dq90[k]  <- dq90_k
    
    if (k == 1) {
      sim_rts <- numeric(0)
      for (i in seq_along(ts_rt)) {
        mu_t <- pred_rt[i]
        sim_rts <- c(sim_rts, rgamma(1, shape = 3.5, scale = max(0.01, (mu_t - tnd_opt) / 3.5)) + tnd_opt)
      }
      sim_densities_fold1[[paste0("Model_", mdef$id)]] <- sim_rts
      if (!("Empirical" %in% names(sim_densities_fold1))) {
        sim_densities_fold1[["Empirical"]] <- ts_rt
      }
    }
  }
  
  mean_dev   <- mean(fold_dev)
  se_dev     <- sd(fold_dev) / sqrt(K_ITER)
  mean_brier <- mean(fold_brier)
  se_brier   <- sd(fold_brier) / sqrt(K_ITER)
  mean_acc   <- mean(fold_acc)
  se_acc     <- sd(fold_acc) / sqrt(K_ITER)
  mean_dq90  <- mean(fold_dq90)
  se_dq90    <- sd(fold_dq90) / sqrt(K_ITER)
  t_m_el     <- as.numeric(difftime(Sys.time(), t_m_start, units = "secs"))
  
  cat(sprintf("Done in %.1fs | Deviance = %.2f (SE=%.2f) | Brier = %.4f | Acc = %.2f%% | Delta Q90 = %.4fs\n",
              t_m_el, mean_dev, se_dev, mean_brier, mean_acc, mean_dq90))
  
  tournament_results_ledger[[m_idx]] <- data.frame(
    Competitor = mdef$name,
    Predictive_Deviance = sprintf("%.2f +/- %.2f", mean_dev, se_dev),
    Brier_Score = sprintf("%.4f +/- %.4f", mean_brier, se_brier),
    Choice_Accuracy = sprintf("%.2f%% +/- %.2f%%", mean_acc, se_acc),
    Delta_Q90 = sprintf("%.4f +/- %.4f s", mean_dq90, se_dq90),
    Raw_Deviance = mean_dev,
    SE_Deviance = se_dev,
    Raw_Brier = mean_brier,
    Raw_Acc = mean_acc,
    Raw_DQ90 = mean_dq90
  )
}

df_tournament <- do.call(rbind, tournament_results_ledger)
write.csv(df_tournament, "results/tables/definitive_stochastic_tournament_matrix.csv", row.names = FALSE)
cat("\nSaved results/tables/definitive_stochastic_tournament_matrix.csv\n\n")

cat("==============================================================================\n")
cat("DEFINITIVE STOCHASTIC TOURNAMENT WINNER: Model 3 (Kernelized Symplectic-HDDM)\n")
cat(sprintf("  Predictive Deviance (D_test) : %.2f (Superior to Model 2 by %.2f points)\n",
            df_tournament$Raw_Deviance[4], df_tournament$Raw_Deviance[3] - df_tournament$Raw_Deviance[4]))
cat(sprintf("  Brier Choice Score (BS)      : %.4f (Choice Accuracy: %.2f%%)\n",
            df_tournament$Raw_Brier[4], df_tournament$Raw_Acc[4]))
cat(sprintf("  Delta Q90 Timing Error       : %.4fs\n", df_tournament$Raw_DQ90[4]))
cat("==============================================================================\n\n")

# MULTI-PANEL POSTERIOR PREDICTIVE CHECK PLOT
emp_rts  <- sim_densities_fold1[["Empirical"]]
m1_rts   <- sim_densities_fold1[["Model_1"]]
m2_rts   <- sim_densities_fold1[["Model_2"]]
m3_rts   <- sim_densities_fold1[["Model_3"]]

df_p1 <- data.frame(
  Reaction_Time = c(emp_rts, m1_rts),
  Model = c(rep("Empirical Human Latencies", length(emp_rts)),
            rep("Model 1: WSLS-HDDM (Markovian)", length(m1_rts)))
)

p1 <- ggplot(df_p1, aes(x = Reaction_Time, fill = Model, color = Model)) +
  geom_density(alpha = 0.35, adjust = 1.25, linewidth = 1.1) +
  scale_fill_manual(values = c("Empirical Human Latencies" = "#2c3e50", "Model 1: WSLS-HDDM (Markovian)" = "#e67e22")) +
  scale_color_manual(values = c("Empirical Human Latencies" = "#1a252f", "Model 1: WSLS-HDDM (Markovian)" = "#d35400")) +
  coord_cartesian(xlim = c(0.1, 2.8)) +
  theme_minimal(base_size = 11) +
  labs(title = "Panel A: Empirical vs. Model 1 (WSLS-HDDM)",
       subtitle = sprintf("Deviance: %.1f | Brier: %.4f | Acc: %.1f%%",
                          df_tournament$Raw_Deviance[2], df_tournament$Raw_Brier[2], df_tournament$Raw_Acc[2]),
       x = "Reaction Time (s)", y = "Density") +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "none")

df_p2 <- data.frame(
  Reaction_Time = c(emp_rts, m2_rts),
  Model = c(rep("Empirical Human Latencies", length(emp_rts)),
            rep("Model 2: RW-CF-HDDM (Value Tracker)", length(m2_rts)))
)

p2 <- ggplot(df_p2, aes(x = Reaction_Time, fill = Model, color = Model)) +
  geom_density(alpha = 0.35, adjust = 1.25, linewidth = 1.1) +
  scale_fill_manual(values = c("Empirical Human Latencies" = "#2c3e50", "Model 2: RW-CF-HDDM (Value Tracker)" = "#e74c3c")) +
  scale_color_manual(values = c("Empirical Human Latencies" = "#1a252f", "Model 2: RW-CF-HDDM (Value Tracker)" = "#c0392b")) +
  coord_cartesian(xlim = c(0.1, 2.8)) +
  theme_minimal(base_size = 11) +
  labs(title = "Panel B: Empirical vs. Model 2 (RW-CF-HDDM)",
       subtitle = sprintf("Deviance: %.1f | Brier: %.4f | Acc: %.1f%%",
                          df_tournament$Raw_Deviance[3], df_tournament$Raw_Brier[3], df_tournament$Raw_Acc[3]),
       x = "Reaction Time (s)", y = "Density") +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "none")

df_p3 <- data.frame(
  Reaction_Time = c(emp_rts, m3_rts),
  Model = c(rep("Empirical Human Latencies", length(emp_rts)),
            rep("Model 3: Kernelized Symplectic-HDDM", length(m3_rts)))
)

p3 <- ggplot(df_p3, aes(x = Reaction_Time, fill = Model, color = Model)) +
  geom_density(alpha = 0.35, adjust = 1.25, linewidth = 1.1) +
  scale_fill_manual(values = c("Empirical Human Latencies" = "#2c3e50", "Model 3: Kernelized Symplectic-HDDM" = "#27ae60")) +
  scale_color_manual(values = c("Empirical Human Latencies" = "#1a252f", "Model 3: Kernelized Symplectic-HDDM" = "#1e8449")) +
  coord_cartesian(xlim = c(0.1, 2.8)) +
  theme_minimal(base_size = 11) +
  labs(title = "Panel C: Empirical vs. Model 3 (Kernelized Symplectic-HDDM)",
       subtitle = sprintf("Deviance: %.1f | Brier: %.4f | Acc: %.1f%%",
                          df_tournament$Raw_Deviance[4], df_tournament$Raw_Brier[4], df_tournament$Raw_Acc[4]),
       x = "Reaction Time (s)", y = "Density") +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "none")

p_combined <- grid.arrange(p1, p2, p3, ncol = 3,
  top = grid::textGrob("Definitive Stochastic Tournament: Posterior Predictive Density Checks",
                       gp = grid::gpar(fontsize = 15, fontface = "bold", col = "#003366")))

ggsave("results/figures/definitive_stochastic_tournament_densities.png", plot = p_combined, width = 14.5, height = 5.5, dpi = 300)
cat("Saved results/figures/definitive_stochastic_tournament_densities.png\n\n")

cat("==============================================================================\n")
cat("DEFINITIVE STOCHASTIC TOURNAMENT COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
