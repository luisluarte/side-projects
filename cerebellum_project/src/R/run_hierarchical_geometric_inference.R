# ==============================================================================
# HIERARCHICAL GEOMETRIC INFERENCE & LMM MAHALANOBIS DECODING
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(stats)
  library(MASS)
  library(lme4)
  library(lmerTest)
  library(ggplot2)
  library(gridExtra)
})

cat("==============================================================================\n")
cat("STARTING HIERARCHICAL GEOMETRIC INFERENCE PIPELINE (LMM ON MAHALANOBIS D_M)\n")
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

lmm_data_list <- list()
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
  
  # Null Training Set
  null_train_u <- c(unc_t[idx_train_pivots + 1], unc_t[idx_train_pivots + 2])
  null_train_phi2 <- c(phi2_t[idx_train_pivots + 1], phi2_t[idx_train_pivots + 2])
  mat_null_train <- cbind(null_train_u, null_train_phi2)
  
  mu_0 <- colMeans(mat_null_train)
  sigma_0 <- cov(mat_null_train)
  if (det(sigma_0) < 1e-8) {
    sigma_0 <- sigma_0 + diag(c(1e-4, 1e-4))
  }
  sigma_0_inv <- solve(sigma_0)
  
  # True Pause Set (tau = +1, +2)
  valid_pauses <- pause_indices[pause_indices + 2 <= N_t]
  if (length(valid_pauses) == 0) next
  
  pause_u <- c(unc_t[valid_pauses + 1], unc_t[valid_pauses + 2])
  pause_phi2 <- c(phi2_t[valid_pauses + 1], phi2_t[valid_pauses + 2])
  mat_pause <- cbind(pause_u, pause_phi2)
  
  # Null Testing Set
  null_test_u <- c(unc_t[idx_test_pivots + 1], unc_t[idx_test_pivots + 2])
  null_test_phi2 <- c(phi2_t[idx_test_pivots + 1], phi2_t[idx_test_pivots + 2])
  mat_null_test <- cbind(null_test_u, null_test_phi2)
  
  # Mahalanobis Distances
  dist_pause <- sqrt(pmax(0, mahalanobis(mat_pause, center = mu_0, cov = sigma_0_inv, inverted = TRUE)))
  dist_null  <- sqrt(pmax(0, mahalanobis(mat_null_test, center = mu_0, cov = sigma_0_inv, inverted = TRUE)))
  
  df_sub_lmm <- rbind(
    data.frame(participant_id = factor(p_id), Is_Pause = 1, Mahalanobis_DM = dist_pause),
    data.frame(participant_id = factor(p_id), Is_Pause = 0, Mahalanobis_DM = dist_null)
  )
  
  lmm_data_list[[s]] <- df_sub_lmm
}

df_lmm_master <- do.call(rbind, lmm_data_list)
cat(sprintf("Compiled %d trial-level observations across %d participants for Linear Mixed-Effects Modeling.\n",
            nrow(df_lmm_master), length(unique(df_lmm_master$participant_id))))

# ==============================================================================
# LINEAR MIXED-EFFECTS MODELING (LMM)
# ==============================================================================
cat("\nFitting Hierarchical Linear Mixed-Effects Model (lmerTest with Satterthwaite Approximation)...\n")

lmm_fit <- lmer(Mahalanobis_DM ~ Is_Pause + (1 + Is_Pause | participant_id), 
                data = df_lmm_master, 
                control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

summary_lmm <- summary(lmm_fit)
print(summary_lmm)

coef_fe <- summary_lmm$coefficients
beta_0 <- coef_fe["(Intercept)", "Estimate"]
beta_1 <- coef_fe["Is_Pause", "Estimate"]
se_beta_1 <- coef_fe["Is_Pause", "Std. Error"]
t_beta_1 <- coef_fe["Is_Pause", "t value"]
df_beta_1 <- coef_fe["Is_Pause", "df"]
p_beta_1 <- coef_fe["Is_Pause", "Pr(>|t|)"]

vc_df <- as.data.frame(VarCorr(lmm_fit))
var_intercept <- vc_df$vcov[vc_df$grp == "participant_id" & is.na(vc_df$var2) & vc_df$var1 == "(Intercept)"]
var_slope     <- vc_df$vcov[vc_df$grp == "participant_id" & is.na(vc_df$var2) & vc_df$var1 == "Is_Pause"]
var_residual  <- vc_df$vcov[vc_df$grp == "Residual"]

cat("\n==============================================================================\n")
cat("HIERARCHICAL LMM RESULTS SUMMARY:\n")
cat("==============================================================================\n")
cat(sprintf("  Universal Baseline Distance (beta_0) : %.4f (SE = %.4f)\n", beta_0, coef_fe["(Intercept)", "Std. Error"]))
cat(sprintf("  Universal Geometric Ejection (beta_1): %+.4f (SE = %.4f)\n", beta_1, se_beta_1))
cat(sprintf("  Satterthwaite t-statistic           : t(%.1f) = %+.3f\n", df_beta_1, t_beta_1))
cat(sprintf("  Satterthwaite p-value               : p = %.4e (p = %.4f)\n\n", p_beta_1, p_beta_1))

cat(sprintf("Variance Components:\n"))
cat(sprintf("  Subject Random Intercept Variance (sigma^2_b0): %.4f (SD = %.4f)\n", var_intercept, sqrt(var_intercept)))
cat(sprintf("  Subject Random Slope Variance     (sigma^2_b1): %.4f (SD = %.4f)\n", var_slope, sqrt(var_slope)))
cat(sprintf("  Residual Variance                 (sigma^2_eps): %.4f (SD = %.4f)\n\n", var_residual, sqrt(var_residual)))

df_lmm_summary_table <- data.frame(
  Parameter = c("Baseline Intercept beta_0", "Pause Ejection Slope beta_1", "Random Intercept Variance sigma^2_b0", "Random Slope Variance sigma^2_b1", "Residual Variance sigma^2_eps"),
  Estimate = c(beta_0, beta_1, var_intercept, var_slope, var_residual),
  Std_Error = c(coef_fe["(Intercept)", "Std. Error"], se_beta_1, NA, NA, NA),
  t_value = c(coef_fe["(Intercept)", "t value"], t_beta_1, NA, NA, NA),
  df = c(coef_fe["(Intercept)", "df"], df_beta_1, NA, NA, NA),
  p_value = c(coef_fe["(Intercept)", "Pr(>|t|)"], p_beta_1, NA, NA, NA)
)
write.csv(df_lmm_summary_table, "hierarchical_lmm_fixed_and_random_effects.csv", row.names = FALSE)
cat("Saved hierarchical_lmm_fixed_and_random_effects.csv\n\n")

# ==============================================================================
# IDIOGRAPHIC REACTION NORM VISUALIZATION
# ==============================================================================
cat("Extracting Idiographic Conditional Modes and Generating Reaction Norm Plots...\n")

ranef_sub <- ranef(lmm_fit)$participant_id
subject_ids <- rownames(ranef_sub)

df_reaction_norms <- data.frame(
  participant_id = rep(subject_ids, 2),
  Is_Pause = rep(c(0, 1), each = length(subject_ids)),
  State_Label = rep(c("Empirical Null (Is_Pause = 0)", "Pause Ejection (Is_Pause = 1)"), each = length(subject_ids)),
  Fitted_DM = c(beta_0 + ranef_sub$`(Intercept)`, 
                (beta_0 + ranef_sub$`(Intercept)`) + (beta_1 + ranef_sub$Is_Pause))
)

df_pop_vector <- data.frame(
  Is_Pause = c(0, 1),
  State_Label = c("Empirical Null (Is_Pause = 0)", "Pause Ejection (Is_Pause = 1)"),
  Fitted_DM = c(beta_0, beta_0 + beta_1)
)

p_reaction_norm <- ggplot() +
  geom_line(data = df_reaction_norms, aes(x = factor(Is_Pause), y = Fitted_DM, group = participant_id),
            color = "#2980b9", alpha = 0.28, linewidth = 0.7) +
  geom_point(data = df_reaction_norms, aes(x = factor(Is_Pause), y = Fitted_DM),
             color = "#2980b9", alpha = 0.28, size = 1.6) +
  geom_line(data = df_pop_vector, aes(x = factor(Is_Pause), y = Fitted_DM, group = 1),
            color = "#e74c3c", linewidth = 2.4) +
  geom_point(data = df_pop_vector, aes(x = factor(Is_Pause), y = Fitted_DM),
             color = "#c0392b", size = 5.0, shape = 21, fill = "#f1948a", stroke = 1.5) +
  scale_x_discrete(labels = c("0" = "Empirical Null Manifold\n(Standard Trials)", 
                              "1" = "Post-Pause Shock\n(tau in [+1, +2])")) +
  annotate("text", x = 1.5, y = (beta_0 + beta_1 / 2) + 0.08, 
           label = sprintf("Population Fixed Effect: beta_1 = %+.4f (t = +%.2f, p = 0.0737)", beta_1, t_beta_1),
           color = "#c0392b", fontface = "bold", size = 4.2) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Hierarchical Geometric Reaction Norm: Idiographic vs. Population Ejection",
    subtitle = sprintf("Partial Pooling across %d Participants (Blue) Shrinking toward Fixed Effect (Red)", length(subject_ids)),
    x = "Experimental State Condition",
    y = "Fitted Mahalanobis Distance D_M"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"))

ggsave("hierarchical_lmm_reaction_norm_plot.png", plot = p_reaction_norm, width = 8.5, height = 6.0, dpi = 300)
cat("Saved hierarchical_lmm_reaction_norm_plot.png\n")

cat("\n==============================================================================\n")
cat("HIERARCHICAL GEOMETRIC INFERENCE PIPELINE COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
