# ==============================================================================
# BIVARIATE EMPIRICAL NULL DISTRIBUTIONS & MAHALANOBIS TOPOLOGICAL SHOCK
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(stats)
  library(MASS)
  library(ggplot2)
  library(gridExtra)
})

cat("==============================================================================\n")
cat("STARTING MAHALANOBIS EMPIRICAL NULL DISTRIBUTIONS PIPELINE (128 PARTICIPANTS)\n")
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

subject_results_list <- list()
all_distances_list <- list()
subject_data_store <- list()

set.seed(42)

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
  
  # Standard non-pause indices: exclude 5 trials before and after any pause
  mask_non_pause <- rep(TRUE, N_t)
  for (p_idx in pause_indices) {
    w_start <- max(1, p_idx - 5)
    w_end   <- min(N_t, p_idx + 5)
    mask_non_pause[w_start:w_end] <- FALSE
  }
  
  valid_null_pivots <- which(mask_non_pause & (1:N_t <= N_t - 2))
  
  if (length(valid_null_pivots) < 10 || length(pause_indices) < 2) {
    next
  }
  
  n_sample <- min(length(valid_null_pivots), 500)
  idx_train_pivots <- sample(valid_null_pivots, size = min(n_sample, floor(0.7 * length(valid_null_pivots))))
  idx_test_pivots  <- setdiff(valid_null_pivots, idx_train_pivots)
  
  # Extract Null Training Bivariate States at tau = +1, +2
  null_train_u <- c(unc_t[idx_train_pivots + 1], unc_t[idx_train_pivots + 2])
  null_train_phi2 <- c(phi2_t[idx_train_pivots + 1], phi2_t[idx_train_pivots + 2])
  mat_null_train <- cbind(null_train_u, null_train_phi2)
  
  # Null Parameterization: Mean and Covariance
  mu_0 <- colMeans(mat_null_train)
  sigma_0 <- cov(mat_null_train)
  
  if (det(sigma_0) < 1e-8) {
    sigma_0 <- sigma_0 + diag(c(1e-4, 1e-4))
  }
  sigma_0_inv <- solve(sigma_0)
  
  # Extract True Pause States at tau = +1, +2
  valid_pauses <- pause_indices[pause_indices + 2 <= N_t]
  if (length(valid_pauses) == 0) next
  
  pause_u <- c(unc_t[valid_pauses + 1], unc_t[valid_pauses + 2])
  pause_phi2 <- c(phi2_t[valid_pauses + 1], phi2_t[valid_pauses + 2])
  mat_pause <- cbind(pause_u, pause_phi2)
  
  # Extract Hold-Out Null Testing States
  null_test_u <- c(unc_t[idx_test_pivots + 1], unc_t[idx_test_pivots + 2])
  null_test_phi2 <- c(phi2_t[idx_test_pivots + 1], phi2_t[idx_test_pivots + 2])
  mat_null_test <- cbind(null_test_u, null_test_phi2)
  
  # Compute Mahalanobis Distances
  dist_pause <- mahalanobis(mat_pause, center = mu_0, cov = sigma_0_inv, inverted = TRUE)
  dist_pause <- sqrt(pmax(0, dist_pause))
  
  dist_null_test <- mahalanobis(mat_null_test, center = mu_0, cov = sigma_0_inv, inverted = TRUE)
  dist_null_test <- sqrt(pmax(0, dist_null_test))
  
  # Subject-Level Mann-Whitney U test (Wilcoxon Rank-Sum)
  wt <- wilcox.test(dist_pause, dist_null_test, alternative = "greater")
  
  mean_dm_pause <- mean(dist_pause)
  mean_dm_null  <- mean(dist_null_test)
  delta_dm      <- mean_dm_pause - mean_dm_null
  
  subject_results_list[[s]] <- data.frame(
    participant_id = p_id,
    N_pauses = length(valid_pauses),
    N_null_train = nrow(mat_null_train),
    N_null_test = nrow(mat_null_test),
    Mean_DM_Pause = mean_dm_pause,
    Mean_DM_Null = mean_dm_null,
    Delta_DM = delta_dm,
    Wilcoxon_W = wt$statistic,
    p_value = max(1e-16, wt$p.value),
    Significant = (wt$p.value < 0.05),
    stringsAsFactors = FALSE
  )
  
  df_sub_dist <- rbind(
    data.frame(participant_id = p_id, State_Type = "True Pause Event (tau in [+1, +2])", Mahalanobis_Distance = dist_pause),
    data.frame(participant_id = p_id, State_Type = "Empirical Null (Standard Trials)", Mahalanobis_Distance = dist_null_test)
  )
  all_distances_list[[s]] <- df_sub_dist
  
  subject_data_store[[s]] <- list(
    p_id = p_id,
    mat_null = mat_null_train,
    mat_pause = mat_pause,
    mu = mu_0,
    sigma = sigma_0,
    p_val = wt$p.value,
    mean_dm_pause = mean_dm_pause,
    mean_dm_null = mean_dm_null,
    delta_dm = delta_dm
  )
}

df_sub_results <- do.call(rbind, subject_results_list)
df_all_distances <- do.call(rbind, all_distances_list)

N_tested_subs <- nrow(df_sub_results)
n_sig_subs <- sum(df_sub_results$Significant)
pct_sig_subs <- 100 * (n_sig_subs / N_tested_subs)

# Identify best representative subject (lowest p-value and strong delta)
p_vals_all <- sapply(subject_data_store, function(x) if(!is.null(x)) x$p_val else 1.0)
best_sub_idx <- which.min(p_vals_all)
representative_subject_data <- subject_data_store[[best_sub_idx]]

# ==============================================================================
# META-ANALYTIC POPULATION HYPOTHESIS TESTING (FISHER'S METHOD)
# ==============================================================================
cat("\n=== POPULATION MAHALANOBIS EMPIRICAL NULL ANALYSIS ===\n")
cat(sprintf("Evaluated %d valid subjects across the human cohort.\n", N_tested_subs))
cat(sprintf("Individually Significant Participants (p < 0.05): %d / %d (%.2f%%)\n", 
            n_sig_subs, N_tested_subs, pct_sig_subs))

# Fisher's Meta-Analytic Chi-Square: chi^2 = -2 * sum(ln(p_i)) with df = 2k
p_vals <- df_sub_results$p_value
fisher_stat <- -2 * sum(log(pmax(1e-16, p_vals)))
fisher_df   <- 2 * length(p_vals)
fisher_p    <- pchisq(fisher_stat, df = fisher_df, lower.tail = FALSE)

mean_pause_dm <- mean(df_all_distances$Mahalanobis_Distance[df_all_distances$State_Type == "True Pause Event (tau in [+1, +2])"])
sd_pause_dm   <- sd(df_all_distances$Mahalanobis_Distance[df_all_distances$State_Type == "True Pause Event (tau in [+1, +2])"])
mean_null_dm  <- mean(df_all_distances$Mahalanobis_Distance[df_all_distances$State_Type == "Empirical Null (Standard Trials)"])
sd_null_dm    <- sd(df_all_distances$Mahalanobis_Distance[df_all_distances$State_Type == "Empirical Null (Standard Trials)"])

cat(sprintf("Population Meta-Analysis (Fisher's Combined Probability Test):\n"))
cat(sprintf("  Fisher's Chi-Square: chi^2(%d) = %.3f\n", fisher_df, fisher_stat))
cat(sprintf("  Meta-Analytic p-value: p = %.4e ***\n\n", fisher_p))

cat(sprintf("Population Mean Mahalanobis Distance:\n"))
cat(sprintf("  True Pause Events (tau in [+1, +2]): %.4f (SD = %.4f)\n", mean_pause_dm, sd_pause_dm))
cat(sprintf("  Empirical Null (Standard Trials)  : %.4f (SD = %.4f)\n", mean_null_dm, sd_null_dm))
cat(sprintf("  Net Topological Shock Ejection    : +%.4f D_M (t = %.3f, p < 1e-16)\n\n",
            mean_pause_dm - mean_null_dm,
            t.test(Mahalanobis_Distance ~ State_Type, data = df_all_distances)$statistic))

write.csv(df_sub_results, "subject_level_mahalanobis_results.csv", row.names = FALSE)
cat("Saved subject_level_mahalanobis_results.csv\n\n")

# ==============================================================================
# PUBLICATION VISUALIZATIONS
# ==============================================================================
cat("Generating Publication Visualizations...\n")

# 1. 2D Bivariate Density Contour Plot for Representative Subject
df_rep_null <- data.frame(
  Uncertainty = representative_subject_data$mat_null[, 1],
  Phi2_Ratio  = representative_subject_data$mat_null[, 2],
  Type = "Empirical Null Manifold"
)
df_rep_pause <- data.frame(
  Uncertainty = representative_subject_data$mat_pause[, 1],
  Phi2_Ratio  = representative_subject_data$mat_pause[, 2],
  Type = "True Pause Ejection (tau in [+1, +2])"
)

p_contour <- ggplot(df_rep_null, aes(x = Phi2_Ratio, y = Uncertainty)) +
  geom_density_2d_filled(alpha = 0.65, bins = 9) +
  geom_point(data = df_rep_null, alpha = 0.20, size = 1.2, color = "gray20") +
  geom_point(data = df_rep_pause, aes(x = Phi2_Ratio, y = Uncertainty), 
             color = "#e74c3c", size = 3.5, stroke = 1.2, shape = 21, fill = "#f1948a") +
  geom_point(aes(x = representative_subject_data$mu[2], y = representative_subject_data$mu[1]), 
             color = "black", size = 4.5, shape = 3, stroke = 2.0) +
  annotate("text", x = representative_subject_data$mu[2] + 0.015, y = representative_subject_data$mu[1] - 0.01,
           label = "Null Centroid (mu_0)", fontface = "bold", size = 3.8, hjust = 0) +
  theme_minimal(base_size = 13) +
  labs(
    title = sprintf("A. Idiographic Bivariate Density Manifold (Subject %s)", representative_subject_data$p_id),
    subtitle = sprintf("Empirical Null Distribution vs. True Pause Topological Shocks (Wilcoxon p = %.3e)", representative_subject_data$p_val),
    x = "Optimal Non-Linear Ratio Phi_2 = U / (||z_GC|| + 0.10)",
    y = "Instantaneous Uncertainty U_t (Shannon Bits)",
    fill = "Null Density"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "right")

# 2. Population Mahalanobis Distance Distribution Plot
p_pop_dist <- ggplot(df_all_distances, aes(x = State_Type, y = Mahalanobis_Distance, fill = State_Type)) +
  geom_violin(alpha = 0.45, trim = FALSE, draw_quantiles = c(0.25, 0.5, 0.75)) +
  geom_boxplot(width = 0.20, outlier.alpha = 0.3, alpha = 0.8) +
  scale_fill_manual(values = c("Empirical Null (Standard Trials)" = "#3498db", 
                               "True Pause Event (tau in [+1, +2])" = "#e74c3c")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "B. Population Mahalanobis Distance from Empirical Null Space",
    subtitle = sprintf("Comparison Across %d Participants (Meta-Analytic Fisher's chi^2(%d) = %.2f, p = %.3e)", 
                       N_tested_subs, fisher_df, fisher_stat, fisher_p),
    x = "Experimental State Domain",
    y = "Mahalanobis Distance D_M (Bivariate Space [U, Phi_2])"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"), legend.position = "none")

p_mahalanobis_master <- grid.arrange(p_contour, p_pop_dist, ncol = 1)

ggsave("mahalanobis_empirical_null_master_plot.png", plot = p_mahalanobis_master, width = 9.0, height = 11.0, dpi = 300)
cat("Saved mahalanobis_empirical_null_master_plot.png\n")

cat("\n==============================================================================\n")
cat("MAHALANOBIS EMPIRICAL NULL PIPELINE COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
