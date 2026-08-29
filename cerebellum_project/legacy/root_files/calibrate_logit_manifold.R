# ==============================================================================
# EXACT-R: LOGIT-TRANSFORMED MANIFOLD EXTRACTION & RISK RELAXATION PIPELINE
# Unbounded Latent Space Optimization: \tilde{\Theta} \in \mathbb{R}^5
# Master Coordinate: D(W_in) \in [0.01, 0.20] (50 Steps)
# Risk Penalties Swept: \beta \in {0.01, 0.025, 0.05, 0.10}
# Kinematic Subset: N = 20 Stratified Maximin Latin Hypercube
# ==============================================================================
suppressPackageStartupMessages({
  library(DiceKriging)
  library(lhs)
  library(splines)
  library(ggplot2)
  library(gridExtra)
  library(parallel)
})

cat("==============================================================================\n")
cat("EXACT-R: Unbounded Logit-Transformed Manifold Extraction & Risk Sweep\n")
cat("==============================================================================\n\n")

# 1. Load 10D Dataset & Fit ARD Matérn 5/2 GP Model
dataset_path <- "gp_lhs_dataset.csv"
if (!file.exists(dataset_path)) {
  stop("gp_lhs_dataset.csv not found. Please run gp_joint_surrogate_pipeline.R first.")
}

full_df <- read.csv(dataset_path)
cat(sprintf("Loaded dataset with %d samples.\n", nrow(full_df)))

param_cols <- c("rho_base_mean", "tau_log_mean", "d_in", "d_fb", "d_inh", "lambda_fb",
                "f_base", "A_load", "delta_phi", "sigma_noise")

X_raw <- as.matrix(full_df[, param_cols])
y_raw <- full_df$Fitness_J

bounds_lower <- c(0.01, 0.50, 0.01, 0.01, 0.05, 0.70, 0.10, 1.00, 0.00, 0.01)
bounds_upper <- c(0.50, 3.00, 0.20, 0.10, 0.30, 0.985, 5.00, 10.00, pi, 1.00)
ranges <- bounds_upper - bounds_lower

X_scaled <- t((t(X_raw) - bounds_lower) / ranges)
colnames(X_scaled) <- param_cols

y_mean <- mean(y_raw)
y_sd <- sd(y_raw) + 1e-12
y_scaled <- (y_raw - y_mean) / y_sd

cat("Fitting DiceKriging GP model (matern5_2 ARD)... ")
gp_model <- km(
  formula = ~1,
  design = X_scaled,
  response = y_scaled,
  covtype = "matern5_2",
  optim.method = "BFGS",
  control = list(trace = FALSE)
)
cat("Done.\n")

# 2. Reduced Kinematic Grid: N = 20 Maximin QMC Subset
set.seed(2026)
N_qmc20 <- 20
phi_bounds_lower <- bounds_lower[7:10]
phi_bounds_upper <- bounds_upper[7:10]
phi_ranges <- phi_bounds_upper - phi_bounds_lower

phi_lhs20 <- maximinLHS(N_qmc20, 4)
phi_qmc20 <- t(t(phi_lhs20) * phi_ranges + phi_bounds_lower)
phi_qmc20_scaled <- t((t(phi_qmc20) - phi_bounds_lower) / phi_ranges)

# 3. Define Logit & Sigmoid Transformations
# 5 Dependent Structural Parameters: rho_base_mean (1), tau_log_mean (2), d_fb (4), d_inh (5), lambda_fb (6)
struct_indices <- c(1, 2, 4, 5, 6)
struct_min <- bounds_lower[struct_indices]
struct_max <- bounds_upper[struct_indices]
struct_range <- struct_max - struct_min

inverse_logit <- function(tilde_theta) {
  struct_min + struct_range / (1 + exp(-tilde_theta))
}

forward_logit <- function(theta) {
  # Clamp slightly inside to avoid log(0)
  theta_c <- pmax(pmin(theta, struct_max - 1e-6), struct_min + 1e-6)
  log((theta_c - struct_min) / (struct_max - theta_c))
}

# 4. Master Grid & Risk Penalty Sweep Parameters
N_steps <- 50
d_in_grid <- seq(0.01, 0.20, length.out = N_steps)
beta_levels <- c(0.01, 0.025, 0.05, 0.10)
gamma_cont <- 2.0  # Smooth homotopy continuation regularization weight in latent space

# Function to run sweep for a single beta
run_logit_sweep <- function(beta_val) {
  # Forward Sweep
  fwd_tilde <- matrix(0, nrow = N_steps, ncol = 5)
  tilde_prev <- rep(0, 5) # Midpoint warm start
  
  for (s in 1:N_steps) {
    x_s <- d_in_grid[s]
    
    obj_fn <- function(tilde_theta) {
      theta_bio <- inverse_logit(tilde_theta)
      theta_full <- c(theta_bio[1], theta_bio[2], x_s, theta_bio[3], theta_bio[4], theta_bio[5])
      theta_scaled <- (theta_full - bounds_lower[1:6]) / ranges[1:6]
      
      eval_matrix <- cbind(
        matrix(rep(theta_scaled, each = N_qmc20), nrow = N_qmc20),
        phi_qmc20_scaled
      )
      colnames(eval_matrix) <- param_cols
      
      pred <- predict(gp_model, newdata = eval_matrix, type = "UK", checkNames = FALSE)
      mu_star <- pred$mean * y_sd + y_mean
      
      mean_mu <- mean(mu_star)
      var_mu <- var(mu_star)
      
      u_marg <- mean_mu - beta_val * var_mu
      penalty <- gamma_cont * sum((tilde_theta - tilde_prev)^2)
      
      return(u_marg - penalty)
    }
    
    opt <- optim(par = tilde_prev, fn = obj_fn, method = "BFGS", control = list(fnscale = -1, maxit = 200))
    fwd_tilde[s, ] <- opt$par
    tilde_prev <- opt$par
  }
  
  # Backward Sweep
  bwd_tilde <- matrix(0, nrow = N_steps, ncol = 5)
  tilde_prev <- fwd_tilde[N_steps, ] # Warm start from forward end
  
  for (s in N_steps:1) {
    x_s <- d_in_grid[s]
    
    obj_fn <- function(tilde_theta) {
      theta_bio <- inverse_logit(tilde_theta)
      theta_full <- c(theta_bio[1], theta_bio[2], x_s, theta_bio[3], theta_bio[4], theta_bio[5])
      theta_scaled <- (theta_full - bounds_lower[1:6]) / ranges[1:6]
      
      eval_matrix <- cbind(
        matrix(rep(theta_scaled, each = N_qmc20), nrow = N_qmc20),
        phi_qmc20_scaled
      )
      colnames(eval_matrix) <- param_cols
      
      pred <- predict(gp_model, newdata = eval_matrix, type = "UK", checkNames = FALSE)
      mu_star <- pred$mean * y_sd + y_mean
      
      mean_mu <- mean(mu_star)
      var_mu <- var(mu_star)
      
      u_marg <- mean_mu - beta_val * var_mu
      penalty <- gamma_cont * sum((tilde_theta - tilde_prev)^2)
      
      return(u_marg - penalty)
    }
    
    opt <- optim(par = tilde_prev, fn = obj_fn, method = "BFGS", control = list(fnscale = -1, maxit = 200))
    bwd_tilde[s, ] <- opt$par
    tilde_prev <- opt$par
  }
  
  # Average latent trajectories & project to biological space
  avg_tilde <- (fwd_tilde + bwd_tilde) / 2.0
  bio_mat <- t(apply(avg_tilde, 1, inverse_logit))
  colnames(bio_mat) <- c("rho_base_mean", "tau_log_mean", "d_fb", "d_inh", "lambda_fb")
  
  # Fit cubic polynomial for each parameter
  r2_cubic <- numeric(5)
  r2_spline <- numeric(5)
  cubic_models <- list()
  
  for (k in 1:5) {
    y_k <- bio_mat[, k]
    fit_cub <- lm(y_k ~ d_in_grid + I(d_in_grid^2) + I(d_in_grid^3))
    r2_cubic[k] <- summary(fit_cub)$r.squared
    cubic_models[[k]] <- fit_cub
    
    fit_spl <- lm(y_k ~ ns(d_in_grid, df = 4))
    r2_spline[k] <- summary(fit_spl)$r.squared
  }
  names(r2_cubic) <- colnames(bio_mat)
  
  return(list(
    beta = beta_val,
    bio_mat = bio_mat,
    r2_cubic = r2_cubic,
    r2_spline = r2_spline,
    min_r2_cubic = min(r2_cubic),
    all_pass = all(r2_cubic > 0.98),
    cubic_models = cubic_models
  ))
}

cat("Executing Risk Sweep across 4 Beta levels in parallel (4 workers)... ")
cl <- makeCluster(min(4, detectCores()))
clusterExport(cl, c("full_df", "param_cols", "X_scaled", "y_scaled", "y_mean", "y_sd",
                    "bounds_lower", "bounds_upper", "ranges", "gp_model",
                    "N_qmc20", "phi_qmc20_scaled", "struct_indices", "struct_min", "struct_max", "struct_range",
                    "inverse_logit", "forward_logit", "N_steps", "d_in_grid", "gamma_cont", "run_logit_sweep"))
clusterEvalQ(cl, {
  library(DiceKriging)
  library(splines)
})

results_list <- parLapply(cl, beta_levels, run_logit_sweep)
stopCluster(cl)
cat("Done!\n\n")

# 5. Evaluate Results & Select Winning Beta
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-8s | %-12s | %-12s | %-12s | %-12s | %-12s | %-10s | %-8s\n",
            "Beta", "R2_rho", "R2_tau", "R2_dfb", "R2_dinh", "R2_lambda", "Min_R2", "Status"))
cat("------------------------------------------------------------------------------\n")

winning_idx <- NULL
min_r2_all <- -Inf

for (i in seq_along(results_list)) {
  res <- results_list[[i]]
  r2 <- res$r2_cubic
  status <- if (res$all_pass) "PASS (>0.98)" else "FAIL"
  
  cat(sprintf("%-8.3f | %-12.4f | %-12.4f | %-12.4f | %-12.4f | %-12.4f | %-10.4f | %-8s\n",
              res$beta, r2[1], r2[2], r2[3], r2[4], r2[5], res$min_r2_cubic, status))
  
  if (res$all_pass && is.null(winning_idx)) {
    winning_idx <- i  # Select minimum beta that passes all > 0.98
  }
}

if (is.null(winning_idx)) {
  cat("\nWARNING: No beta met R2_cubic > 0.98 for all 5 parameters simultaneously.\n")
  cat("Selecting beta that maximizes the minimum R2_cubic across parameters...\n")
  max_min_r2 <- -Inf
  for (i in seq_along(results_list)) {
    if (results_list[[i]]$min_r2_cubic > max_min_r2) {
      max_min_r2 <- results_list[[i]]$min_r2_cubic
      winning_idx <- i
    }
  }
}

winning_res <- results_list[[winning_idx]]
cat("------------------------------------------------------------------------------\n\n")
cat("==============================================================================\n")
cat("WINNING LOGIT FORMULATION IDENTIFIED:\n")
cat(sprintf("  Selected Beta:      %.3f\n", winning_res$beta))
cat(sprintf("  Analytical BCR:     0.0000 (Guaranteed 0%% Clamping)\n"))
cat(sprintf("  Minimum R2_cubic:   %.4f\n", winning_res$min_r2_cubic))
cat("==============================================================================\n\n")

# 6. Extract Cubic Polynomial Coefficients for C++ Struct
coeffs_list <- list()
for (k in 1:5) {
  fit <- winning_res$cubic_models[[k]]
  cf <- coef(fit)
  coeffs_list[[colnames(winning_res$bio_mat)[k]]] <- cf
}

# Function to format C++ inline functions
format_cpp_func <- function(func_name, cf) {
  sprintf("  static inline double %s(double d_in) {\n    return %.6f + (%.6f * d_in) + (%.6f * d_in * d_in) + (%.6f * d_in * d_in * d_in);\n  }",
          func_name, cf[1], cf[2], cf[3], cf[4])
}

cpp_rho <- format_cpp_func("get_rho_base_mean", coeffs_list[["rho_base_mean"]])
cpp_tau <- format_cpp_func("get_tau_log_mean", coeffs_list[["tau_log_mean"]])
cpp_dfb <- format_cpp_func("get_d_fb", coeffs_list[["d_fb"]])
cpp_dinh <- format_cpp_func("get_d_inh", coeffs_list[["d_inh"]])
cpp_lambda <- format_cpp_func("get_lambda_fb", coeffs_list[["lambda_fb"]])

cat("==============================================================================\n")
cat("REVISED C++ HARDCODABLE SMOOTH RIDGE MANIFOLD STRUCT (Copy-Paste Ready)\n")
cat("==============================================================================\n")
cat("// Calibrated Unbounded Logit Ridge Manifold (Beta = ", sprintf("%.3f", winning_res$beta), ", BCR = 0.0000)\n", sep = "")
cat("// Higher-Order Cubic Algebraic Polynomial Models\n")
cat("struct SmoothRidgeManifold {\n")
cat(cpp_rho, "\n\n")
cat(cpp_tau, "\n\n")
cat(cpp_dfb, "\n\n")
cat(cpp_dinh, "\n\n")
cat(cpp_lambda, "\n")
cat("};\n")
cat("==============================================================================\n\n")

# 7. Verification of lambda_fb at 0.01 and 0.20
cf_lam <- coeffs_list[["lambda_fb"]]
eval_lambda <- function(x) cf_lam[1] + cf_lam[2]*x + cf_lam[3]*x^2 + cf_lam[4]*x^3

val_001 <- eval_lambda(0.01)
val_020 <- eval_lambda(0.20)

cat("==============================================================================\n")
cat("EXPLICIT SPECTRAL RADIUS BOUNDARY VERIFICATION:\n")
cat(sprintf("  get_lambda_fb(0.01) = %.6f  (Requirement: <= 0.99) -> %s\n",
            val_001, if (val_001 <= 0.99) "VERIFIED PASS" else "FAIL"))
cat(sprintf("  get_lambda_fb(0.20) = %.6f  (Requirement: <= 0.99) -> %s\n",
            val_020, if (val_020 <= 0.99) "VERIFIED PASS" else "FAIL"))
cat("==============================================================================\n\n")

# Save ledger & plot
ledger_df <- data.frame(
  Beta = sapply(results_list, function(x) x$beta),
  BCR = 0.0,
  R2_rho = sapply(results_list, function(x) x$r2_cubic[1]),
  R2_tau = sapply(results_list, function(x) x$r2_cubic[2]),
  R2_dfb = sapply(results_list, function(x) x$r2_cubic[3]),
  R2_dinh = sapply(results_list, function(x) x$r2_cubic[4]),
  R2_lambda = sapply(results_list, function(x) x$r2_cubic[5]),
  Min_R2 = sapply(results_list, function(x) x$min_r2_cubic),
  Passed_098 = sapply(results_list, function(x) x$all_pass)
)
write.csv(ledger_df, "logit_calibration_ledger.csv", row.names = FALSE)
cat("Saved logit calibration ledger to logit_calibration_ledger.csv\n")

# Plot diagnostic fits
plot_df <- data.frame(d_in = d_in_grid, winning_res$bio_mat)
p1 <- ggplot(plot_df, aes(x = d_in, y = rho_base_mean)) + geom_point(color = "blue") + geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "red") + theme_minimal() + labs(title = "rho_base_mean vs D(W_in)")
p2 <- ggplot(plot_df, aes(x = d_in, y = tau_log_mean)) + geom_point(color = "blue") + geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "red") + theme_minimal() + labs(title = "tau_log_mean vs D(W_in)")
p3 <- ggplot(plot_df, aes(x = d_in, y = d_fb)) + geom_point(color = "blue") + geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "red") + theme_minimal() + labs(title = "d_fb vs D(W_in)")
p4 <- ggplot(plot_df, aes(x = d_in, y = d_inh)) + geom_point(color = "blue") + geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "red") + theme_minimal() + labs(title = "d_inh vs D(W_in)")
p5 <- ggplot(plot_df, aes(x = d_in, y = lambda_fb)) + geom_point(color = "blue") + geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "red") + theme_minimal() + labs(title = "lambda_fb vs D(W_in)")

g <- grid.arrange(p1, p2, p3, p4, p5, ncol = 3, top = sprintf("Calibrated Unbounded Logit Ridge Manifold (Beta = %.3f)", winning_res$beta))
ggsave("logit_calibration_fits.png", g, width = 12, height = 8)
cat("Saved diagnostic plots to logit_calibration_fits.png\n")
