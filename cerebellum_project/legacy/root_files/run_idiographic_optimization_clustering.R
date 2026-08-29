# ==============================================================================
# IDIOGRAPHIC PARAMETER OPTIMIZATION & PHENOTYPIC CLUSTERING (128 PARTICIPANTS)
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(stats)
  library(PRROC)
  library(pROC)
  library(ggplot2)
})

cat("==============================================================================\n")
cat("STARTING IDIOGRAPHIC (SUBJECT-LEVEL) PARAMETER OPTIMIZATION\n")
cat("==============================================================================\n\n")

sourceCpp("ExactRModel.cpp")

dataset_path <- "../datasets/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

participants <- unique(dat_all[['participant_id']])
N_sub <- length(participants)

cat(sprintf("Loaded %d participants with %d valid trials.\n", N_sub, nrow(dat_all)))

# Parameter bounds for Theta in R^10
lower_bounds <- c(0.50, 0.10, 0.00, 0.00, 0.01, 0.00, 0.00, 0.50, 0.00, 0.00)
upper_bounds <- c(0.99, 0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 0.99, 0.50, 0.50)
param_names <- c("p_ws_base", "p_ls_base", "w_mag_curr", "w_mag_alt", "alpha_q", 
                 "w_streak", "w_purkinje_inh", "tau_kinematic", "beta_post_err", "kappa_entropy")

theta_pop <- matrix(0, nrow = N_sub, ncol = 10)
colnames(theta_pop) <- param_names
rownames(theta_pop) <- participants

wsls_alignment <- numeric(N_sub)
wsls_nll_vec <- numeric(N_sub)
model_nll_vec <- numeric(N_sub)
model_rt_rmse_vec <- numeric(N_sub)
model_prauc_vec <- numeric(N_sub)

# Optimization loop per subject
for (s in 1:N_sub) {
  p_id <- participants[s]
  sub_df <- dat_all[dat_all[['participant_id']] == p_id, ]
  resp <- as.numeric(sub_df[['Resp']])
  out <- as.numeric(sub_df[['F']])
  m1 <- as.numeric(sub_df[['Bd1']])
  m2 <- as.numeric(sub_df[['Bd2']])
  rt <- as.numeric(sub_df[['RT']])
  N_t <- length(resp)
  
  # 1. Compute Subject WSLS Baseline
  # Count Win-Stay and Lose-Shift transitions
  prev_resp <- resp[1:(N_t - 1)]
  prev_out  <- out[1:(N_t - 1)]
  curr_resp <- resp[2:N_t]
  
  win_trials  <- which(prev_out == 1)
  lose_trials <- which(prev_out == 0)
  
  p_ws_emp <- if(length(win_trials) > 0) mean(curr_resp[win_trials] == prev_resp[win_trials]) else 0.5
  p_ls_emp <- if(length(lose_trials) > 0) mean(curr_resp[lose_trials] != prev_resp[lose_trials]) else 0.5
  
  p_ws_emp <- max(0.01, min(0.99, p_ws_emp))
  p_ls_emp <- max(0.01, min(0.99, p_ls_emp))
  
  # WSLS NLL
  nll_wsls <- 0
  for (t in 2:N_t) {
    if (prev_out[t - 1] == 1) {
      p_c <- if (curr_resp[t - 1] == prev_resp[t - 1]) p_ws_emp else (1 - p_ws_emp)
    } else {
      p_c <- if (curr_resp[t - 1] != prev_resp[t - 1]) p_ls_emp else (1 - p_ls_emp)
    }
    nll_wsls <- nll_wsls - log(max(1e-12, p_c))
  }
  wsls_nll_vec[s] <- nll_wsls
  
  # Strict WSLS transition fraction (Alignment metric)
  strict_wsls_count <- sum(curr_resp[win_trials] == prev_resp[win_trials]) + 
                       sum(curr_resp[lose_trials] != prev_resp[lose_trials])
  wsls_alignment[s] <- strict_wsls_count / (N_t - 1)
  
  # 2. Optimize exact cortico-cerebellar parameters for subject s
  obj_fn <- function(th_uncon) {
    # Constrain to Theta manifold via logistic transform
    th <- lower_bounds + (upper_bounds - lower_bounds) / (1 + exp(-th_uncon))
    res <- run_exact_r_simulation_cpp(resp, out, m1, m2, rt, th)
    
    # Joint Loss: Choice NLL + 20 * RT RMSE
    rt_err <- as.numeric(res$RT_Emp) - as.numeric(res$RT_Preds)
    rmse_rt <- sqrt(mean(rt_err^2, na.rm = TRUE))
    
    loss <- res$Choice_NLL + 20.0 * rmse_rt
    return(loss)
  }
  
  # Initial guess centered around population priors
  init_th <- c(p_ws_emp, p_ls_emp, 0.20, 0.15, 0.28, 0.08, 0.10, 0.96, 0.04, 0.03)
  init_uncon <- -log((upper_bounds - lower_bounds) / (pmax(lower_bounds + 1e-4, pmin(upper_bounds - 1e-4, init_th)) - lower_bounds) - 1)
  
  opt <- optim(init_uncon, obj_fn, method = "Nelder-Mead", control = list(maxit = 350))
  
  th_opt <- lower_bounds + (upper_bounds - lower_bounds) / (1 + exp(-opt$par))
  theta_pop[s, ] <- th_opt
  
  # Evaluate optimal subject model
  res_opt <- run_exact_r_simulation_cpp(resp, out, m1, m2, rt, th_opt)
  model_nll_vec[s] <- res_opt$Choice_NLL
  
  rt_e <- as.numeric(res_opt$RT_Emp)
  rt_p <- as.numeric(res_opt$RT_Preds)
  model_rt_rmse_vec[s] <- sqrt(mean((rt_e - rt_p)^2, na.rm = TRUE))
  
  lbls <- as.numeric(res_opt$Switch_Labels)
  prbs <- as.numeric(res_opt$Switch_Probs)
  clean <- !is.na(lbls) & !is.na(prbs)
  if (sum(lbls[clean] == 1) > 0 && sum(lbls[clean] == 0) > 0) {
    pr <- pr.curve(scores.class0 = prbs[clean & lbls == 1], scores.class1 = prbs[clean & lbls == 0], curve = FALSE)
    model_prauc_vec[s] <- pr[['auc.integral']]
  } else {
    model_prauc_vec[s] <- 0.50
  }
}

cat("Optimization completed across all 128 participants!\n\n")

# Save Population Parameter Matrix
df_pop <- as.data.frame(theta_pop)
df_pop$participant_id <- participants
df_pop$WSLS_Alignment <- wsls_alignment
df_pop$WSLS_NLL <- wsls_nll_vec
df_pop$Model_NLL <- model_nll_vec
df_pop$Model_RT_RMSE <- model_rt_rmse_vec
df_pop$Model_PR_AUC <- model_prauc_vec

write.csv(df_pop, "idiographic_population_parameter_matrix.csv", row.names = FALSE)
cat("Saved idiographic_population_parameter_matrix.csv\n\n")

# ==============================================================================
# TOPOLOGICAL ANALYSIS & PHENOTYPE MAPPING
# ==============================================================================
cat("Performing Dimensionality Reduction (PCA & Topological Clustering)...\n")

pca_res <- prcomp(theta_pop, center = TRUE, scale. = TRUE)
df_pop$PC1 <- pca_res$x[, 1]
df_pop$PC2 <- pca_res$x[, 2]
var_explained <- summary(pca_res)$importance[2, 1:2] * 100

cat(sprintf("PC1 explains %.2f%% of variance | PC2 explains %.2f%% of variance.\n", 
            var_explained[1], var_explained[2]))

# Spearman Correlation with WSLS Alignment
spearman_corrs <- numeric(10)
names(spearman_corrs) <- param_names
p_vals <- numeric(10)
names(p_vals) <- param_names

for (p in param_names) {
  test_res <- cor.test(df_pop[[p]], df_pop$WSLS_Alignment, method = "spearman")
  spearman_corrs[p] <- test_res$estimate
  p_vals[p] <- test_res$p.value
}

cat("\nSpearman Correlation between Parameters and WSLS Alignment:\n")
for (p in param_names) {
  cat(sprintf("  %-16s: rho = %+.4f (p = %.4e)\n", p, spearman_corrs[p], p_vals[p]))
}

df_corrs <- data.frame(
  Parameter = param_names,
  Spearman_Rho = spearman_corrs,
  P_Value = p_vals
)
write.csv(df_corrs, "wsls_alignment_parameter_correlations.csv", row.names = FALSE)

# Multiple Regression
fit_lm <- lm(WSLS_Alignment ~ p_ws_base + p_ls_base + w_mag_curr + w_mag_alt + 
               alpha_q + w_streak + w_purkinje_inh + tau_kinematic + beta_post_err + kappa_entropy, 
             data = df_pop)
cat("\nMultiple Linear Regression Summary:\n")
print(summary(fit_lm))

# Plot PCA Topological Landscape colored by WSLS Alignment
p_topo <- ggplot(df_pop, aes(x = PC1, y = PC2, color = WSLS_Alignment)) +
  geom_point(size = 3.5, alpha = 0.85) +
  scale_color_gradientn(colors = c("#003366", "#0088cc", "#2eb872", "#f1c40f", "#e74c3c"),
                        name = "WSLS\nAlignment") +
  theme_minimal(base_size = 14) +
  labs(
    title = "Topological Manifold of Idiographic Cerebellar Phenotypes",
    subtitle = sprintf("PCA Projection of 128 Human Subjects (%d Trials) colored by WSLS Heuristic Alignment", nrow(dat_all)),
    x = sprintf("Principal Component 1 (%.1f%% Variance)", var_explained[1]),
    y = sprintf("Principal Component 2 (%.1f%% Variance)", var_explained[2])
  ) +
  theme(
    plot.title = element_text(face = "bold", color = "#003366"),
    legend.position = "right"
  )

ggsave("phenotypic_landscape_pca_plot.png", plot = p_topo, width = 8.5, height = 5.5, dpi = 300)
cat("Saved phenotypic_landscape_pca_plot.png\n")

# Plot Parameter Correlation Bar Plot
df_corrs_plot <- df_corrs[order(df_corrs$Spearman_Rho), ]
df_corrs_plot$Parameter <- factor(df_corrs_plot$Parameter, levels = df_corrs_plot$Parameter)

p_bar <- ggplot(df_corrs_plot, aes(x = Parameter, y = Spearman_Rho, fill = Spearman_Rho > 0)) +
  geom_bar(stat = "identity", width = 0.65) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#2eb872", "FALSE" = "#e74c3c"), guide = "none") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Cerebellar Biophysical Drivers of Discrete WSLS Heuristic Alignment",
    subtitle = "Spearman Correlation (rho) between 10-D Parameters and Empirical WSLS Alignment",
    x = "Biophysical Parameter",
    y = "Spearman Correlation (rho) with WSLS Alignment"
  ) +
  theme(plot.title = element_text(face = "bold", color = "#003366"))

ggsave("wsls_parameter_correlation_bar_plot.png", plot = p_bar, width = 8.5, height = 5.0, dpi = 300)
cat("Saved wsls_parameter_correlation_bar_plot.png\n")

cat("\n==============================================================================\n")
cat("IDIOGRAPHIC ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
