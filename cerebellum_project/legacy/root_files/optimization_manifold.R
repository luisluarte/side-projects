# ==============================================================================
# EXACT-R: Optimization Manifold & Scree/SVD Elbow Criterion Analysis
# ==============================================================================
library(dplyr)

# Automated Elbow Detection using Maximum Curvature (Kneedle Method)
detect_scree_elbow <- function(eigenvalues) {
  N <- length(eigenvalues)
  if (N < 3) return(1)
  
  # Normalize eigenvalues to [0, 1]
  y <- (eigenvalues - min(eigenvalues)) / (max(eigenvalues) - min(eigenvalues) + 1e-12)
  x <- seq(0, 1, length.out = N)
  
  p1 <- c(x[1], y[1])
  p2 <- c(x[N], y[N])
  
  distances <- numeric(N)
  for (i in 1:N) {
    p0 <- c(x[i], y[i])
    numerator <- abs((p2[2] - p1[2]) * p0[1] - (p2[1] - p1[1]) * p0[2] + p2[1] * p1[2] - p2[2] * p1[1])
    denominator <- sqrt((p2[2] - p1[2])^2 + (p2[1] - p1[1])^2) + 1e-12
    distances[i] <- numerator / denominator
  }
  
  elbow_idx <- which.max(distances)
  return(elbow_idx)
}

# Main function to derive the Optimization Manifold
derive_optimization_manifold <- function(sweep_file = "optimization_sweep_results.csv",
                                         sens_file = "sensitivity_analysis_results.csv",
                                         target_protocol = "Filtered") {
  
  cat("\n==============================================================================\n")
  cat("DERIVING OPTIMIZATION MANIFOLD WITH AUTOMATED ELBOW CRITERION ANALYSIS\n")
  cat("==============================================================================\n")
  
  trials_df <- read.csv(sweep_file, stringsAsFactors = FALSE)
  if (file.exists(sens_file)) {
    sens_df <- read.csv(sens_file, stringsAsFactors = FALSE)
    trials_df <- bind_rows(trials_df, sens_df)
  }
  
  trials_df <- na.omit(trials_df)
  
  prot_trials <- trials_df %>% filter(Protocol == target_protocol)
  if (nrow(prot_trials) < 5) {
    prot_trials <- trials_df
  }
  
  param_cols <- c("rho_base_mean", "rho_base_sd", "tau_log_mean", "tau_log_sd",
                  "d_in", "d_fb", "d_inh", "d_collateral", "lambda_fb")
  
  param_cols <- intersect(param_cols, colnames(prot_trials))
  param_matrix <- as.matrix(prot_trials[, param_cols])
  
  # Sort trials by Fitness_J descending
  sorted_idx <- order(prot_trials$Fitness_J, decreasing = TRUE)
  sorted_params <- param_matrix[sorted_idx, , drop = FALSE]
  sorted_fitness <- prot_trials$Fitness_J[sorted_idx]
  
  # Add small jitter to columns with near-zero variance
  for (j in 1:ncol(sorted_params)) {
    if (sd(sorted_params[, j]) < 1e-8) {
      sorted_params[, j] <- sorted_params[, j] + rnorm(nrow(sorted_params), mean = 0, sd = 1e-6)
    }
  }
  
  # 1. Automated Elbow Detection on Parameter Covariance Eigenvalues
  param_scaled <- scale(sorted_params)
  param_cov <- cov(param_scaled)
  
  eig_res <- eigen(param_cov)
  eig_cov <- pmax(0, Re(eig_res$values))
  eig_var_explained <- eig_cov / sum(eig_cov)
  
  elbow_rank <- detect_scree_elbow(eig_cov)
  cat(sprintf("Scree Plot Eigenvalue Elbow Rank Detected: %d / %d (Var Explained: %.2f%%)\n",
              elbow_rank, length(eig_cov), sum(eig_var_explained[1:elbow_rank]) * 100))
  
  # 2. Automated Elbow Detection on Fitness Progression Curve
  fit_elbow_idx <- detect_scree_elbow(sorted_fitness)
  top_n_elbow <- max(5, fit_elbow_idx)
  
  cat(sprintf("Fitness Progression Curve Elbow Index Detected: %d trials selected (Top %.1f%%)\n",
              top_n_elbow, (top_n_elbow / nrow(prot_trials)) * 100))
  
  elbow_params <- sorted_params[1:top_n_elbow, , drop = FALSE]
  
  # Ensure all columns in elbow_params have non-zero variance for PCA
  for (j in 1:ncol(elbow_params)) {
    if (sd(elbow_params[, j]) < 1e-8) {
      elbow_params[, j] <- elbow_params[, j] + rnorm(nrow(elbow_params), mean = 0, sd = 1e-6)
    }
  }
  
  # 3. Principal Component Analysis (PCA) on Elbow-Filtered Parameter Manifold
  pca_res <- prcomp(elbow_params, scale. = TRUE)
  pca_var_explained <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
  
  cat("\n--- Principal Components Variance Explained ---\n")
  for (pc in 1:min(5, length(pca_var_explained))) {
    cat(sprintf("PC %d: %6.2f%% (Cumulative: %6.2f%%)\n",
                pc, pca_var_explained[pc] * 100, sum(pca_var_explained[1:pc]) * 100))
  }
  
  # 4. Multivariable Linear Manifold Equation Derivation
  elbow_df <- as.data.frame(elbow_params)
  
  lm_formula <- as.formula("d_inh ~ d_fb + lambda_fb + rho_base_mean + tau_log_mean")
  lm_inh <- lm(lm_formula, data = elbow_df)
  lm_summary <- summary(lm_inh)
  
  cat("\n==============================================================================\n")
  cat("CLOSED-FORM OPTIMIZATION MANIFOLD EQUATION (Golgi Inhibition Coupling)\n")
  cat("==============================================================================\n")
  cat(sprintf("Model Fit R^2: %.4f (Adjusted R^2: %.4f, p-value: %.4e)\n",
              lm_summary$r.squared, lm_summary$adj.r.squared,
              ifelse(is.null(lm_summary$fstatistic), 1.0, pf(lm_summary$fstatistic[1], lm_summary$fstatistic[2], lm_summary$fstatistic[3], lower.tail = FALSE))))
  
  coeffs <- coef(lm_inh)
  cat("\nLinear Manifold Coefficients:\n")
  for (c_name in names(coeffs)) {
    cat(sprintf("  %-15s = %+.4f\n", c_name, coeffs[c_name]))
  }
  cat("==============================================================================\n")
  
  coeff_df <- data.frame(
    Parameter = names(coeffs),
    Coefficient = as.numeric(coeffs),
    R_Squared = lm_summary$r.squared,
    Elbow_Rank = elbow_rank,
    Elbow_Top_N = top_n_elbow,
    stringsAsFactors = FALSE
  )
  write.csv(coeff_df, "optimization_manifold_coeffs.csv", row.names = FALSE)
  
  return(list(
    Elbow_Rank = elbow_rank,
    Top_N_Elbow = top_n_elbow,
    PCA_Var_Explained = pca_var_explained,
    Linear_Model = lm_inh,
    Coefficients = coeffs
  ))
}
