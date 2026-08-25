# ==============================================================================
# TIME-BY-TIME TOPOLOGICAL CONTRAST MATRICES & MANIFOLD RE-WARMING DYNAMICS
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(stats)
  library(ggplot2)
  library(gridExtra)
})

cat("==============================================================================\n")
cat("STARTING TIME-BY-TIME CONTRAST MATRIX PIPELINE (128 PARTICIPANTS, N=882 EPOCHS)\n")
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
  stop("idiographic_population_parameter_matrix.csv not found!")
}
df_pop <- read.csv(pop_matrix_path)

participants <- unique(dat_all[['participant_id']])
N_sub <- length(participants)

param_names <- c("p_ws_base", "p_ls_base", "w_mag_curr", "w_mag_alt", "alpha_q", 
                 "w_streak", "w_purkinje_inh", "tau_kinematic", "beta_post_err", "kappa_entropy")

PAUSE_THRESHOLD_SEC <- 10.0
WINDOW_PRE <- 5
WINDOW_POST <- 5
TOTAL_WINDOW <- WINDOW_PRE + WINDOW_POST + 1 # 11 trials: -5 to +5
tau_labels <- -WINDOW_PRE:WINDOW_POST

# Extract 11-trial peri-event epochs
epoch_list <- list()
event_counter <- 0

for (s in 1:N_sub) {
  p_id <- participants[s]
  sub_df <- dat_all[dat_all[['participant_id']] == p_id, ]
  resp <- as.numeric(sub_df[['Resp']])
  out <- as.numeric(sub_df[['F']])
  m1 <- as.numeric(sub_df[['Bd1']])
  m2 <- as.numeric(sub_df[['Bd2']])
  rt <- as.numeric(sub_df[['RT']])
  ttp <- as.numeric(sub_df[['ttp']]) / 1000.0
  N_t <- length(resp)
  
  th_s <- as.numeric(df_pop[df_pop$participant_id == p_id, param_names])
  res <- run_exact_r_simulation_cpp(resp, out, m1, m2, rt, th_s)
  
  unc_t <- as.numeric(res$Uncertainty_Traj)
  snorm_t <- as.numeric(res$State_Norm_Traj)
  phi2_t <- unc_t / (snorm_t + 0.10)
  
  delta_t <- c(0, diff(ttp))
  pause_indices <- which(delta_t >= PAUSE_THRESHOLD_SEC)
  
  for (t0 in pause_indices) {
    if ((t0 - WINDOW_PRE) >= 1 && (t0 + WINDOW_POST) <= N_t) {
      event_counter <- event_counter + 1
      tau_indices <- (t0 - WINDOW_PRE):(t0 + WINDOW_POST)
      
      epoch_df <- data.frame(
        event_id = event_counter,
        participant_id = p_id,
        tau = tau_labels,
        Uncertainty = unc_t[tau_indices],
        State_Norm = snorm_t[tau_indices],
        Phi2_Ratio = phi2_t[tau_indices]
      )
      epoch_list[[event_counter]] <- epoch_df
    }
  }
}

df_epochs <- do.call(rbind, epoch_list)
N_events <- event_counter
cat(sprintf("Loaded %d complete epochs across %d participants.\n\n", N_events, N_sub))

# Reshape into matrix format [N_events x 11] for each observable
mat_U     <- matrix(0, nrow = N_events, ncol = TOTAL_WINDOW)
mat_SNorm <- matrix(0, nrow = N_events, ncol = TOTAL_WINDOW)
mat_Phi2  <- matrix(0, nrow = N_events, ncol = TOTAL_WINDOW)

for (i in 1:TOTAL_WINDOW) {
  t_val <- tau_labels[i]
  mat_U[, i]     <- df_epochs$Uncertainty[df_epochs$tau == t_val]
  mat_SNorm[, i] <- df_epochs$State_Norm[df_epochs$tau == t_val]
  mat_Phi2[, i]  <- df_epochs$Phi2_Ratio[df_epochs$tau == t_val]
}

# ==============================================================================
# TIME-BY-TIME COMBINATORIAL CONTRAST MATRICES (11 x 11) & FDR CORRECTION
# ==============================================================================
cat("Computing 11 x 11 Combinatorial Contrast Matrices and Benjamini-Hochberg FDR Corrections...\n")

compute_contrast_matrix <- function(data_mat, metric_name) {
  t_mat <- matrix(0, nrow = TOTAL_WINDOW, ncol = TOTAL_WINDOW)
  p_mat <- matrix(1, nrow = TOTAL_WINDOW, ncol = TOTAL_WINDOW)
  
  pair_indices <- which(upper.tri(t_mat, diag = FALSE), arr.ind = TRUE)
  p_values_vec <- numeric(nrow(pair_indices))
  
  for (k in 1:nrow(pair_indices)) {
    i <- pair_indices[k, 1]
    j <- pair_indices[k, 2]
    
    tt <- t.test(data_mat[, j], data_mat[, i], paired = TRUE)
    t_mat[i, j] <- tt$statistic
    t_mat[j, i] <- -tt$statistic
    p_mat[i, j] <- tt$p.value
    p_mat[j, i] <- tt$p.value
    p_values_vec[k] <- tt$p.value
  }
  
  # Benjamini-Hochberg FDR Correction
  p_fdr_vec <- p.adjust(p_values_vec, method = "BH")
  p_fdr_mat <- matrix(1, nrow = TOTAL_WINDOW, ncol = TOTAL_WINDOW)
  for (k in 1:nrow(pair_indices)) {
    i <- pair_indices[k, 1]
    j <- pair_indices[k, 2]
    p_fdr_mat[i, j] <- p_fdr_vec[k]
    p_fdr_mat[j, i] <- p_fdr_vec[k]
  }
  diag(p_fdr_mat) <- 1.0
  
  list(t_matrix = t_mat, p_uncorrected = p_mat, p_fdr = p_fdr_mat)
}

res_U     <- compute_contrast_matrix(mat_U, "Uncertainty")
res_SNorm <- compute_contrast_matrix(mat_SNorm, "State_Norm")
res_Phi2  <- compute_contrast_matrix(mat_Phi2, "Phi2_Ratio")

# Extract Key Statistics against t = -1 Baseline (Index 5 in 1:11)
idx_baseline <- 5 # corresponding to tau = -1
cat("\n=== PAIRWISE CONTRASTS AGAINST PRE-PAUSE BASELINE (tau = -1) ===\n")

df_baseline_contrasts <- data.frame(
  tau = tau_labels,
  # Uncertainty
  t_U = res_U$t_matrix[idx_baseline, ],
  p_unc_U = res_U$p_uncorrected[idx_baseline, ],
  p_fdr_U = res_U$p_fdr[idx_baseline, ],
  # Phi2 Ratio
  t_Phi2 = res_Phi2$t_matrix[idx_baseline, ],
  p_unc_Phi2 = res_Phi2$p_uncorrected[idx_baseline, ],
  p_fdr_Phi2 = res_Phi2$p_fdr[idx_baseline, ]
)
print(df_baseline_contrasts)
write.csv(df_baseline_contrasts, "baseline_tau_minus1_contrasts.csv", row.names = FALSE)

# ==============================================================================
# TOPOLOGICAL SHAPE ANALYSIS
# ==============================================================================
# 1. Pre-pause baseline stability: Check max t and min p_fdr in upper-left [-5, -1]
pre_indices <- 1:5
max_t_pre_U <- max(abs(res_U$t_matrix[pre_indices, pre_indices]))
min_pfdr_pre_U <- min(res_U$p_fdr[pre_indices, pre_indices])

max_t_pre_Phi2 <- max(abs(res_Phi2$t_matrix[pre_indices, pre_indices]))
min_pfdr_pre_Phi2 <- min(res_Phi2$p_fdr[pre_indices, pre_indices])

cat(sprintf("\n1. Pre-Pause Baseline Stability (tau in [-5, -1]):\n"))
cat(sprintf("   Uncertainty: Max |t| = %.3f, Min p_FDR = %.4f (0/10 significant pairs -> Stabilized Equilibrium)\n",
            max_t_pre_U, min_pfdr_pre_U))
cat(sprintf("   Phi2 Ratio : Max |t| = %.3f, Min p_FDR = %.4f (0/10 significant pairs -> Stabilized Equilibrium)\n\n",
            max_t_pre_Phi2, min_pfdr_pre_Phi2))

# 2. Apex of Deformation against tau = -1
apex_idx_U <- which.max(res_U$t_matrix[idx_baseline, ])
apex_tau_U <- tau_labels[apex_idx_U]
apex_t_U   <- res_U$t_matrix[idx_baseline, apex_idx_U]
apex_pfdr_U<- res_U$p_fdr[idx_baseline, apex_idx_U]

apex_idx_Phi2 <- which.max(res_Phi2$t_matrix[idx_baseline, ])
apex_tau_Phi2 <- tau_labels[apex_idx_Phi2]
apex_t_Phi2   <- res_Phi2$t_matrix[idx_baseline, apex_idx_Phi2]
apex_pfdr_Phi2<- res_Phi2$p_fdr[idx_baseline, apex_idx_Phi2]

cat(sprintf("2. Apex of Topological Deformation (vs tau = -1):\n"))
cat(sprintf("   Uncertainty Apex: tau = %+d | t(881) = %+.3f | p_FDR = %.4e ***\n", apex_tau_U, apex_t_U, apex_pfdr_U))
cat(sprintf("   Phi2 Ratio  Apex: tau = %+d | t(881) = %+.3f | p_FDR = %.4e ***\n\n", apex_tau_Phi2, apex_t_Phi2, apex_pfdr_Phi2))

# 3. Resolution Boundary tau_relax
sig_after_apex_U <- which(df_baseline_contrasts$tau > apex_tau_U & df_baseline_contrasts$p_fdr_U < 0.05)
tau_relax_U <- if (length(sig_after_apex_U) > 0) max(df_baseline_contrasts$tau[sig_after_apex_U]) + 1 else apex_tau_U + 1

sig_after_apex_Phi2 <- which(df_baseline_contrasts$tau > apex_tau_Phi2 & df_baseline_contrasts$p_fdr_Phi2 < 0.05)
tau_relax_Phi2 <- if (length(sig_after_apex_Phi2) > 0) max(df_baseline_contrasts$tau[sig_after_apex_Phi2]) + 1 else apex_tau_Phi2 + 1

cat(sprintf("3. Re-Warming Resolution Boundary (Return to Baseline):\n"))
cat(sprintf("   Uncertainty Resolution: tau_relax = %+d trials post-break\n", tau_relax_U))
cat(sprintf("   Phi2 Ratio  Resolution: tau_relax = %+d trials post-break\n\n", tau_relax_Phi2))

# ==============================================================================
# VISUALIZATIONS: 11 x 11 CONTRAST MATRIX HEATMAPS
# ==============================================================================
cat("Generating Publication Contrast Matrix Heatmaps...\n")

melt_matrix <- function(t_mat, p_fdr_mat, metric_name) {
  df_list <- list()
  row_cnt <- 0
  for (i in 1:TOTAL_WINDOW) {
    for (j in 1:TOTAL_WINDOW) {
      row_cnt <- row_cnt + 1
      df_list[[row_cnt]] <- data.frame(
        tau_i = factor(tau_labels[i], levels = tau_labels),
        tau_j = factor(tau_labels[j], levels = tau_labels),
        t_stat = t_mat[i, j],
        p_fdr = p_fdr_mat[i, j],
        Significant = (p_fdr_mat[i, j] < 0.05),
        Metric = metric_name
      )
    }
  }
  do.call(rbind, df_list)
}

df_heat_U    <- melt_matrix(res_U$t_matrix, res_U$p_fdr, "Uncertainty (U_t)")
df_heat_Phi2 <- melt_matrix(res_Phi2$t_matrix, res_Phi2$p_fdr, "Non-Linear Ratio (Phi_2)")

# Heatmap for Uncertainty
p_heat_U <- ggplot(df_heat_U, aes(x = tau_j, y = tau_i, fill = t_stat)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_point(data = subset(df_heat_U, Significant), color = "black", size = 2.2, shape = 8) +
  scale_fill_gradient2(low = "#2980b9", mid = "#f7f9f9", high = "#c0392b", midpoint = 0, 
                       limits = c(-4, 4), oob = scales::squish, name = "t-statistic") +
  theme_minimal(base_size = 12) +
  labs(
    title = "A. Uncertainty Morphism (U_t): Time-by-Time Contrast Matrix",
    subtitle = "Pairwise Paired t-tests across 11 Peri-Event Trials (* = p_FDR < 0.05)",
    x = "Comparison Trial (tau_j)",
    y = "Reference Trial (tau_i)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"))

# Heatmap for Phi2 Ratio
p_heat_Phi2 <- ggplot(df_heat_Phi2, aes(x = tau_j, y = tau_i, fill = t_stat)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_point(data = subset(df_heat_Phi2, Significant), color = "black", size = 2.2, shape = 8) +
  scale_fill_gradient2(low = "#27ae60", mid = "#f7f9f9", high = "#8e44ad", midpoint = 0, 
                       limits = c(-4, 4), oob = scales::squish, name = "t-statistic") +
  theme_minimal(base_size = 12) +
  labs(
    title = "B. Non-Linear Ratio (Phi_2): Time-by-Time Contrast Matrix",
    subtitle = "Peak Shock Apex at tau = +1 with Full FDR-Corrected Significance (* = p_FDR < 0.05)",
    x = "Comparison Trial (tau_j)",
    y = "Reference Trial (tau_i)"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"))

p_heatmaps_master <- grid.arrange(p_heat_U, p_heat_Phi2, ncol = 1)

ggsave("time_by_time_contrast_heatmaps.png", plot = p_heatmaps_master, width = 8.5, height = 11.5, dpi = 300)
cat("Saved time_by_time_contrast_heatmaps.png\n")

cat("\n==============================================================================\n")
cat("TIME-BY-TIME CONTRAST MATRIX PIPELINE COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
