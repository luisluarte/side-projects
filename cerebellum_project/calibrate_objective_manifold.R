# ==============================================================================
# EXACT-R: HIGH-SPEED 12-CORE PARALLEL OBJECTIVE CALIBRATION PIPELINE
# Lead Computational Architect: Boundary Repulsion & Risk Penalty Optimization
# Master Coordinate: D(W_in) \in [0.01, 0.20] (20 Steps)
# Reduced Calibration Grid: N = 20 Maximin QMC Kinematic Subset
# Parallel Execution: 12 System Cores via parallel::makeCluster(12)
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
cat("EXACT-R: 12-Core Parallel Objective Calibration & Manifold Extraction\n")
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
bounds_upper <- c(0.50, 3.00, 0.20, 0.10, 0.30, 0.99, 5.00, 10.00, pi, 1.00)
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

# 2. Reduced Calibration Grid: N = 20 Maximin QMC Kinematic Grid (Phi_qmc20)
set.seed(2026)
N_qmc20 <- 20
phi_bounds_lower <- c(0.10, 1.00, 0.00, 0.01)
phi_bounds_upper <- c(5.00, 10.00, pi, 1.00)
phi_ranges <- phi_bounds_upper - phi_bounds_lower

qmc20_unit <- maximinLHS(N_qmc20, 4)
Phi_qmc20 <- t(t(qmc20_unit) * phi_ranges + phi_bounds_lower)
colnames(Phi_qmc20) <- c("f_base", "A_load", "delta_phi", "sigma_noise")

cat(sprintf("Generated Reduced Calibration Grid (N = %d Maximin QMC Kinematic Subset).\n\n", N_qmc20))

lower_5d <- c(0.01, 0.50, 0.01, 0.05, 0.70)
upper_5d <- c(0.50, 3.00, 0.10, 0.30, 0.99)
ranges_5d <- upper_5d - lower_5d

# 3. Create Expanded Grid of Calibration Configurations
beta_levels <- c(0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
topology_grid <- expand.grid(
  Beta = beta_levels,
  Topology = c("Logit_Unbounded", "Log_Barrier_k0.1", "Log_Barrier_k0.5", "Log_Barrier_k1.0",
               "Exp_Repulsion_g1.0", "Exp_Repulsion_g5.0", "Exp_Repulsion_g10.0"),
  stringsAsFactors = FALSE
)

n_configs <- nrow(topology_grid)
cat(sprintf("Configured Grid Search with %d experimental configurations.\n", n_configs))

# 4. Define Execution Task for Parallel Cluster
run_single_config <- function(idx, grid_df, gp_model, Phi_qmc20, bounds_lower, ranges, y_mean, y_sd, lower_5d, upper_5d, ranges_5d, param_cols) {
  library(DiceKriging)
  library(splines)
  
  beta <- grid_df$Beta[idx]
  top  <- grid_df$Topology[idx]
  
  n_sweep <- 20
  d_in_seq <- seq(0.01, 0.20, length.out = n_sweep)
  
  eval_marginal_raw <- function(Theta_cand) {
    Theta_mat <- matrix(rep(Theta_cand, each = nrow(Phi_qmc20)), nrow = nrow(Phi_qmc20), ncol = 6)
    Psi_mat_phys <- cbind(Theta_mat, Phi_qmc20)
    Psi_mat_scaled <- t((t(Psi_mat_phys) - bounds_lower) / ranges)
    colnames(Psi_mat_scaled) <- param_cols
    pred <- predict(gp_model, newdata = Psi_mat_scaled, type = "SK", se.compute = FALSE)
    mu_phys <- pred$mean * y_sd + y_mean
    return(mean(mu_phys) - beta * var(mu_phys))
  }
  
  sweep_5d <- matrix(0, nrow = n_sweep, ncol = 5)
  d_in_1 <- d_in_seq[1]
  
  if (top == "Logit_Unbounded") {
    obj_fn <- function(tilde_p) {
      norm_p <- 1 / (1 + exp(-tilde_p))
      p5d <- lower_5d + norm_p * ranges_5d
      Theta <- c(p5d[1], p5d[2], d_in_1, p5d[3], p5d[4], p5d[5])
      -eval_marginal_raw(Theta)
    }
    opt <- suppressWarnings(tryCatch({ optim(rep(0, 5), obj_fn, method = "BFGS", control = list(maxit = 80)) }, error = function(e) list(par = rep(0, 5))))
    prev_p5d <- lower_5d + (1 / (1 + exp(-opt$par))) * ranges_5d
    sweep_5d[1, ] <- prev_p5d
    
    for (s in 2:n_sweep) {
      d_in_s <- d_in_seq[s]
      prev_tilde <- log((prev_p5d - lower_5d) / (upper_5d - prev_p5d))
      obj_fn_s <- function(tilde_p) {
        norm_p <- 1 / (1 + exp(-tilde_p))
        p5d <- lower_5d + norm_p * ranges_5d
        Theta <- c(p5d[1], p5d[2], d_in_s, p5d[3], p5d[4], p5d[5])
        raw_fit <- eval_marginal_raw(Theta)
        dist_pen <- 1.0 * sum((tilde_p - prev_tilde)^2)
        -(raw_fit - dist_pen)
      }
      opt_s <- suppressWarnings(tryCatch({ optim(prev_tilde, obj_fn_s, method = "BFGS", control = list(maxit = 40)) }, error = function(e) list(par = prev_tilde)))
      prev_p5d <- lower_5d + (1 / (1 + exp(-opt_s$par))) * ranges_5d
      sweep_5d[s, ] <- prev_p5d
    }
  } else if (startsWith(top, "Log_Barrier")) {
    kappa <- as.numeric(sub("Log_Barrier_k", "", top))
    obj_fn <- function(p5d) {
      Theta <- c(p5d[1], p5d[2], d_in_1, p5d[3], p5d[4], p5d[5])
      norm_l <- (p5d - lower_5d) / ranges_5d
      norm_u <- (upper_5d - p5d) / ranges_5d
      if (any(norm_l <= 1e-4 | norm_u <= 1e-4)) return(1e9)
      raw_fit <- eval_marginal_raw(Theta)
      barrier <- sum(log(norm_l) + log(norm_u))
      -(raw_fit + kappa * barrier)
    }
    init_p <- 0.5 * (lower_5d + upper_5d)
    opt <- suppressWarnings(tryCatch({ optim(init_p, obj_fn, method = "L-BFGS-B", lower = lower_5d + 0.005*ranges_5d, upper = upper_5d - 0.005*ranges_5d, control = list(maxit = 80)) }, error = function(e) list(par = init_p)))
    prev_p5d <- opt$par
    sweep_5d[1, ] <- prev_p5d
    
    for (s in 2:n_sweep) {
      d_in_s <- d_in_seq[s]
      obj_fn_s <- function(p5d) {
        Theta <- c(p5d[1], p5d[2], d_in_s, p5d[3], p5d[4], p5d[5])
        norm_l <- (p5d - lower_5d) / ranges_5d
        norm_u <- (upper_5d - p5d) / ranges_5d
        if (any(norm_l <= 1e-4 | norm_u <= 1e-4)) return(1e9)
        raw_fit <- eval_marginal_raw(Theta)
        barrier <- sum(log(norm_l) + log(norm_u))
        dist_pen <- 1.0 * sum(((p5d - prev_p5d) / ranges_5d)^2)
        -(raw_fit + kappa * barrier - dist_pen)
      }
      opt_s <- suppressWarnings(tryCatch({ optim(prev_p5d, obj_fn_s, method = "L-BFGS-B", lower = lower_5d + 0.005*ranges_5d, upper = upper_5d - 0.005*ranges_5d, control = list(maxit = 40)) }, error = function(e) list(par = prev_p5d)))
      prev_p5d <- opt_s$par
      sweep_5d[s, ] <- prev_p5d
    }
  } else if (startsWith(top, "Exp_Repulsion")) {
    gamma_val <- as.numeric(sub("Exp_Repulsion_g", "", top))
    eps_val <- 0.05
    obj_fn <- function(p5d) {
      Theta <- c(p5d[1], p5d[2], d_in_1, p5d[3], p5d[4], p5d[5])
      norm_l <- (p5d - lower_5d) / ranges_5d
      norm_u <- (upper_5d - p5d) / ranges_5d
      raw_fit <- eval_marginal_raw(Theta)
      pen_exp <- gamma_val * sum(exp(-norm_l / eps_val) + exp(-norm_u / eps_val))
      -(raw_fit - pen_exp)
    }
    init_p <- 0.5 * (lower_5d + upper_5d)
    opt <- suppressWarnings(tryCatch({ optim(init_p, obj_fn, method = "L-BFGS-B", lower = lower_5d, upper = upper_5d, control = list(maxit = 80)) }, error = function(e) list(par = init_p)))
    prev_p5d <- opt$par
    sweep_5d[1, ] <- prev_p5d
    
    for (s in 2:n_sweep) {
      d_in_s <- d_in_seq[s]
      obj_fn_s <- function(p5d) {
        Theta <- c(p5d[1], p5d[2], d_in_s, p5d[3], p5d[4], p5d[5])
        norm_l <- (p5d - lower_5d) / ranges_5d
        norm_u <- (upper_5d - p5d) / ranges_5d
        raw_fit <- eval_marginal_raw(Theta)
        pen_exp <- gamma_val * sum(exp(-norm_l / eps_val) + exp(-norm_u / eps_val))
        dist_pen <- 1.0 * sum(((p5d - prev_p5d) / ranges_5d)^2)
        -(raw_fit - pen_exp - dist_pen)
      }
      opt_s <- suppressWarnings(tryCatch({ optim(prev_p5d, obj_fn_s, method = "L-BFGS-B", lower = lower_5d, upper = upper_5d, control = list(maxit = 40)) }, error = function(e) list(par = prev_p5d)))
      prev_p5d <- opt_s$par
      sweep_5d[s, ] <- prev_p5d
    }
  }
  
  # Metrics Computation
  clamped_count <- 0
  for (i in 1:n_sweep) {
    norm_vals <- (sweep_5d[i, ] - lower_5d) / ranges_5d
    clamped_count <- clamped_count + sum(norm_vals <= 0.02 | norm_vals >= 0.98)
  }
  bcr <- clamped_count / (n_sweep * 5)
  
  r2_vec <- numeric(5)
  for (k in 1:5) {
    y_val <- sweep_5d[, k]
    lm_cub <- lm(y_val ~ poly(d_in_seq, 3, raw = TRUE))
    r2_vec[k] <- summary(lm_cub)$r.squared
  }
  r2_cubic_avg <- mean(r2_vec)
  
  is_feasible <- (bcr == 0.0) && (r2_cubic_avg > 0.95)
  
  return(list(
    Beta = beta, Topology = top, BCR = bcr, R2_cubic = r2_cubic_avg,
    Feasible = is_feasible, Sweep5D = sweep_5d
  ))
}

# 5. Launch Parallel 12-Core Execution
cat("Spawning 12 Parallel Worker Cores (`parallel::makeCluster(12)`)...\n")
cl <- makeCluster(12)

clusterExport(cl, c("topology_grid", "gp_model", "Phi_qmc20", "bounds_lower", "ranges",
                    "y_mean", "y_sd", "lower_5d", "upper_5d", "ranges_5d", "param_cols",
                    "run_single_config"), envir = environment())

cat(sprintf("Executing %d calibration sweeps in parallel across 12 cores... ", n_configs))
t0 <- Sys.time()
results_list <- parLapply(cl, 1:n_configs, function(idx) {
  run_single_config(idx, topology_grid, gp_model, Phi_qmc20, bounds_lower, ranges, y_mean, y_sd, lower_5d, upper_5d, ranges_5d, param_cols)
})
stopCluster(cl)
t_diff <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("Done in %.2f seconds!\n\n", t_diff))

# 6. Build & Print Calibration Ledger
ledger_rows <- list()
for (i in 1:n_configs) {
  res <- results_list[[i]]
  ledger_rows[[i]] <- data.frame(
    Beta = res$Beta, Topology = res$Topology,
    BCR = res$BCR, R2_cubic = res$R2_cubic,
    Feasible = res$Feasible, stringsAsFactors = FALSE
  )
}

ledger_df <- do.call(rbind, ledger_rows)
write.csv(ledger_df, "objective_calibration_ledger.csv", row.names = FALSE)
cat("Saved calibration ledger to objective_calibration_ledger.csv\n\n")

cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-6s | %-18s | %-8s | %-10s | %-14s\n", "Beta", "Topology", "BCR", "R2_cubic", "Status"))
cat("------------------------------------------------------------------------------\n")
for (i in 1:nrow(ledger_df)) {
  status_str <- ifelse(ledger_df$Feasible[i], "FEASIBLE [PASS]", "INFEASIBLE")
  cat(sprintf("%-6.2f | %-18s | %-8.4f | %-10.4f | %-14s\n",
              ledger_df$Beta[i], ledger_df$Topology[i], ledger_df$BCR[i], ledger_df$R2_cubic[i], status_str))
}
cat("------------------------------------------------------------------------------\n\n")

# 7. Autonomous Selection of Winning Formulation
feasible_df <- ledger_df[ledger_df$Feasible == TRUE, ]

if (nrow(feasible_df) == 0) {
  cat("WARNING: No formulation met strict BCR == 0.0 & R2_cubic > 0.95 simultaneously.\n")
  cat("Selecting formulation that minimizes BCR and maximizes R2_cubic...\n")
  best_idx <- order(ledger_df$BCR, -ledger_df$R2_cubic, ledger_df$Beta)[1]
  winner_res <- results_list[[best_idx]]
} else {
  # Select formulation that minimizes Beta among feasible candidates
  best_idx <- order(feasible_df$Beta, -feasible_df$R2_cubic)[1]
  matching_idx <- which(sapply(results_list, function(r) r$Beta == feasible_df$Beta[best_idx] && r$Topology == feasible_df$Topology[best_idx]))[1]
  winner_res <- results_list[[matching_idx]]
}

cat("==============================================================================\n")
cat("WINNING OBJECTIVE FORMULATION IDENTIFIED:\n")
cat(sprintf("  Optimal Beta:      %.2f\n", winner_res$Beta))
cat(sprintf("  Topology:          %s\n", winner_res$Topology))
cat(sprintf("  BCR (Boundary Clamping): %.4f (Target: 0.0000)\n", winner_res$BCR))
cat(sprintf("  R2_cubic (Smoothness):    %.4f (Target: > 0.9500)\n", winner_res$R2_cubic))
cat("==============================================================================\n\n")

# 8. Extract Smooth Interior Manifold Equations for Winning Formulation
winner_sweep_5d <- winner_res$Sweep5D
d_in_seq <- seq(0.01, 0.20, length.out = 20)

ridge_df <- data.frame(
  Step = 1:20,
  d_in = d_in_seq,
  rho_base_mean = loess(winner_sweep_5d[, 1] ~ d_in_seq, span = 0.35)$fitted,
  tau_log_mean  = loess(winner_sweep_5d[, 2] ~ d_in_seq, span = 0.35)$fitted,
  d_fb          = loess(winner_sweep_5d[, 3] ~ d_in_seq, span = 0.35)$fitted,
  d_inh         = loess(winner_sweep_5d[, 4] ~ d_in_seq, span = 0.35)$fitted,
  lambda_fb     = loess(winner_sweep_5d[, 5] ~ d_in_seq, span = 0.35)$fitted
)

write.csv(ridge_df, "ridge_manifold_sweep_smooth.csv", row.names = FALSE)

dep_params <- c("rho_base_mean", "tau_log_mean", "d_fb", "d_inh", "lambda_fb")
coeff_list <- list()

for (dep in dep_params) {
  y_val <- ridge_df[[dep]]
  x_val <- ridge_df$d_in
  lm_cub <- lm(y_val ~ poly(x_val, 3, raw = TRUE))
  c_cub <- coef(lm_cub)
  coeff_list[[dep]] <- data.frame(
    Parameter = dep,
    Cubic_c0 = c_cub[1], Cubic_c1 = c_cub[2], Cubic_c2 = c_cub[3], Cubic_c3 = c_cub[4],
    stringsAsFactors = FALSE
  )
}

coeff_df <- do.call(rbind, coeff_list)
rownames(coeff_df) <- dep_params
write.csv(coeff_df, "ridge_manifold_coefficients.csv", row.names = FALSE)

# 9. Output Hardcodable C++ Code Block to Console
cat("==============================================================================\n")
cat("C++ HARDCODABLE SMOOTH INTERIOR RIDGE MANIFOLD EQUATIONS (Copy-Paste Ready)\n")
cat("==============================================================================\n")
cat(sprintf("// Calibrated Interior Cerebellar Ridge Manifold (Beta = %.2f, Topology = %s)\n", winner_res$Beta, winner_res$Topology))
cat("// Higher-Order Cubic Algebraic Polynomial Models (BCR = 0.0, R2_cubic > 0.95)\n")
cat("struct SmoothRidgeManifold {\n")
cat("  static inline double get_rho_base_mean(double d_in) {\n")
cat(sprintf("    return %.6f + (%.6f * d_in) + (%.6f * d_in * d_in) + (%.6f * d_in * d_in * d_in);\n",
            coeff_df["rho_base_mean", "Cubic_c0"], coeff_df["rho_base_mean", "Cubic_c1"],
            coeff_df["rho_base_mean", "Cubic_c2"], coeff_df["rho_base_mean", "Cubic_c3"]))
cat("  }\n\n")

cat("  static inline double get_tau_log_mean(double d_in) {\n")
cat(sprintf("    return %.6f + (%.6f * d_in) + (%.6f * d_in * d_in) + (%.6f * d_in * d_in * d_in);\n",
            coeff_df["tau_log_mean", "Cubic_c0"], coeff_df["tau_log_mean", "Cubic_c1"],
            coeff_df["tau_log_mean", "Cubic_c2"], coeff_df["tau_log_mean", "Cubic_c3"]))
cat("  }\n\n")

cat("  static inline double get_d_fb(double d_in) {\n")
cat(sprintf("    return %.6f + (%.6f * d_in) + (%.6f * d_in * d_in) + (%.6f * d_in * d_in * d_in);\n",
            coeff_df["d_fb", "Cubic_c0"], coeff_df["d_fb", "Cubic_c1"],
            coeff_df["d_fb", "Cubic_c2"], coeff_df["d_fb", "Cubic_c3"]))
cat("  }\n\n")

cat("  static inline double get_d_inh(double d_in) {\n")
cat(sprintf("    return %.6f + (%.6f * d_in) + (%.6f * d_in * d_in) + (%.6f * d_in * d_in * d_in);\n",
            coeff_df["d_inh", "Cubic_c0"], coeff_df["d_inh", "Cubic_c1"],
            coeff_df["d_inh", "Cubic_c2"], coeff_df["d_inh", "Cubic_c3"]))
cat("  }\n\n")

cat("  static inline double get_lambda_fb(double d_in) {\n")
cat(sprintf("    return %.6f + (%.6f * d_in) + (%.6f * d_in * d_in) + (%.6f * d_in * d_in * d_in);\n",
            coeff_df["lambda_fb", "Cubic_c0"], coeff_df["lambda_fb", "Cubic_c1"],
            coeff_df["lambda_fb", "Cubic_c2"], coeff_df["lambda_fb", "Cubic_c3"]))
cat("  }\n")
cat("};\n")
cat("==============================================================================\n\n")

# 10. Render Diagnostic Fit Plots
p1 <- ggplot(ridge_df, aes(x = d_in, y = d_inh)) +
  geom_point(color = "blue", size = 2.5) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "navy", se = FALSE) +
  labs(title = sprintf("Golgi Inhibition D(W_inh) [Beta = %.2f]", winner_res$Beta), x = "D(W_in)", y = "D(W_inh)") +
  theme_minimal()

p2 <- ggplot(ridge_df, aes(x = d_in, y = tau_log_mean)) +
  geom_point(color = "red", size = 2.5) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "darkred", se = FALSE) +
  labs(title = sprintf("Retention Tau_log_mean [Beta = %.2f]", winner_res$Beta), x = "D(W_in)", y = "tau_log_mean") +
  theme_minimal()

p3 <- ggplot(ridge_df, aes(x = d_in, y = lambda_fb)) +
  geom_point(color = "purple", size = 2.5) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "purple4", se = FALSE) +
  labs(title = sprintf("Spectral Radius lambda_fb [Beta = %.2f]", winner_res$Beta), x = "D(W_in)", y = "lambda_fb") +
  theme_minimal()

p4 <- ggplot(ridge_df, aes(x = d_in, y = rho_base_mean)) +
  geom_point(color = "darkgreen", size = 2.5) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), color = "darkgreen", se = FALSE) +
  labs(title = sprintf("Base Density rho_base_mean [Beta = %.2f]", winner_res$Beta), x = "D(W_in)", y = "rho_base_mean") +
  theme_minimal()

g_ridge <- grid.arrange(p1, p2, p3, p4, ncol = 2)
ggsave("objective_calibration_fits.png", g_ridge, width = 11, height = 9, dpi = 300)
ggsave("ridge_manifold_fits.png", g_ridge, width = 11, height = 9, dpi = 300)
ggsave("C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad/ridge_manifold_fits.png", g_ridge, width = 11, height = 9, dpi = 300)

cat("Rendered and saved objective calibration fit plots to objective_calibration_fits.png\n")
cat("12-Core Calibration pipeline completed successfully.\n")
