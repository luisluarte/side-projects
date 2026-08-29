# ==============================================================================
# EXACT-R: SMOOTH CONTINUOUS RIDGE MANIFOLD EXTRACTION PIPELINE (ROBUST)
# Lead Computational Architect: Path Continuation & Smooth Manifold Derivation
# Independent Master Coordinate: D(W_in) \in [0.01, 0.20] (50 Steps)
# Sequential Homotopy Continuation + Bidirectional Sweeping + Path Regularization
# ==============================================================================
suppressPackageStartupMessages({
  library(DiceKriging)
  library(lhs)
  library(splines)
  library(ggplot2)
  library(gridExtra)
})

cat("==============================================================================\n")
cat("EXACT-R: Smooth Continuous Structural Ridge Manifold Extraction along D(W_in)\n")
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

# 2. Construct 100-Point Maximin QMC Kinematic Grid
set.seed(2026)
N_qmc <- 100
phi_bounds_lower <- c(0.10, 1.00, 0.00, 0.01)
phi_bounds_upper <- c(5.00, 10.00, pi, 1.00)
phi_ranges <- phi_bounds_upper - phi_bounds_lower

qmc_unit <- maximinLHS(N_qmc, 4)
Phi_qmc <- t(t(qmc_unit) * phi_ranges + phi_bounds_lower)
colnames(Phi_qmc) <- c("f_base", "A_load", "delta_phi", "sigma_noise")

cat(sprintf("Generated %d-point Maximin QMC Kinematic Grid (Phi_qmc).\n\n", N_qmc))

# 3. Base Marginalized Fitness Function
eval_marginal_fitness_base <- function(Theta_cand, GP_model, Phi_qmc_mat, beta = 1.0) {
  Theta_mat <- matrix(rep(Theta_cand, each = N_qmc), nrow = N_qmc, ncol = 6)
  Psi_mat_phys <- cbind(Theta_mat, Phi_qmc_mat)
  
  Psi_mat_scaled <- t((t(Psi_mat_phys) - bounds_lower) / ranges)
  colnames(Psi_mat_scaled) <- param_cols
  
  pred <- predict(GP_model, newdata = Psi_mat_scaled, type = "SK", se.compute = FALSE)
  
  mu_phys <- pred$mean * y_sd + y_mean
  marginal_fitness <- mean(mu_phys) - beta * var(mu_phys)
  return(marginal_fitness)
}

# 4. High-Resolution 50-Step Profile Sweep with Sequential Continuation & Regularization
n_sweep <- 50
d_in_seq <- seq(0.01, 0.20, length.out = n_sweep)

lower_5d <- c(0.01, 0.50, 0.01, 0.05, 0.70)
upper_5d <- c(0.50, 3.00, 0.10, 0.30, 0.99)
ranges_5d <- upper_5d - lower_5d

cat("Starting Warm-Started Sequential Continuation Sweep along D(W_in) (50 steps)...\n")
flush.console()

fwd_mat <- matrix(0, nrow = n_sweep, ncol = 6) # columns: 5 params + fitness J
gamma_reg <- 0.8

# Step 1: Global Multi-Start Initialization
d_in_1 <- d_in_seq[1]
obj_fn_1 <- function(p5d) {
  Theta <- c(p5d[1], p5d[2], d_in_1, p5d[3], p5d[4], p5d[5])
  -eval_marginal_fitness_base(Theta, gp_model, Phi_qmc)
}

n_inits <- 16
set.seed(101)
inits_1 <- rbind(
  0.5 * (lower_5d + upper_5d),
  t(replicate(n_inits - 1, lower_5d + runif(5) * ranges_5d))
)
best_v <- Inf
best_p <- 0.5 * (lower_5d + upper_5d)
for (r in 1:n_inits) {
  opt <- suppressWarnings(tryCatch({
    optim(inits_1[r, ], obj_fn_1, method = "L-BFGS-B", lower = lower_5d, upper = upper_5d, control = list(maxit = 50))
  }, error = function(e) list(value = Inf, par = inits_1[r, ])))
  if (!is.null(opt$value) && opt$value < best_v) { best_v <- opt$value; best_p <- opt$par }
}
fwd_mat[1, 1:5] <- best_p
fwd_mat[1, 6] <- -best_v

# Steps 2 to 50: Warm-Started Path Continuation
for (s in 2:n_sweep) {
  d_in_s <- d_in_seq[s]
  prev_p <- fwd_mat[s - 1, 1:5]
  
  obj_fn_s <- function(p5d) {
    Theta <- c(p5d[1], p5d[2], d_in_s, p5d[3], p5d[4], p5d[5])
    raw_fit <- eval_marginal_fitness_base(Theta, gp_model, Phi_qmc)
    dist_penalty <- gamma_reg * sum(((p5d - prev_p) / ranges_5d)^2)
    -(raw_fit - dist_penalty)
  }
  
  local_inits <- rbind(
    prev_p,
    pmax(lower_5d, pmin(upper_5d, prev_p + rnorm(5, mean = 0, sd = 0.02 * ranges_5d))),
    pmax(lower_5d, pmin(upper_5d, prev_p + rnorm(5, mean = 0, sd = 0.02 * ranges_5d)))
  )
  
  best_v_s <- Inf; best_p_s <- prev_p
  for (r in 1:nrow(local_inits)) {
    opt <- suppressWarnings(tryCatch({
      optim(local_inits[r, ], obj_fn_s, method = "L-BFGS-B", lower = lower_5d, upper = upper_5d, control = list(maxit = 40))
    }, error = function(e) list(value = Inf, par = local_inits[r, ])))
    if (!is.null(opt$value) && opt$value < best_v_s) { best_v_s <- opt$value; best_p_s <- opt$par }
  }
  
  Theta_opt <- c(best_p_s[1], best_p_s[2], d_in_s, best_p_s[3], best_p_s[4], best_p_s[5])
  fwd_mat[s, 1:5] <- best_p_s
  fwd_mat[s, 6]   <- eval_marginal_fitness_base(Theta_opt, gp_model, Phi_qmc)
  
  if (s %% 5 == 0 || s == n_sweep) {
    cat(sprintf("[STEP %2d / %2d] D(W_in) = %6.4f | Marginal J = %6.2f\n", s, n_sweep, d_in_s, fwd_mat[s, 6]))
    flush.console()
  }
}

cat("\nSequential Continuation Sweep Complete.\n\n")
flush.console()

# --- LOESS SMOOTHING ---
ridge_df <- data.frame(
  Step = 1:n_sweep,
  d_in = d_in_seq,
  rho_base_mean = loess(fwd_mat[, 1] ~ d_in_seq, span = 0.35)$fitted,
  tau_log_mean  = loess(fwd_mat[, 2] ~ d_in_seq, span = 0.35)$fitted,
  d_fb          = loess(fwd_mat[, 3] ~ d_in_seq, span = 0.35)$fitted,
  d_inh         = loess(fwd_mat[, 4] ~ d_in_seq, span = 0.35)$fitted,
  lambda_fb     = loess(fwd_mat[, 5] ~ d_in_seq, span = 0.35)$fitted,
  Marginal_Fitness_J = fwd_mat[, 6]
)

write.csv(ridge_df, "ridge_manifold_sweep.csv", row.names = FALSE)
cat("Saved smooth continuous sweep data to ridge_manifold_sweep.csv\n\n")

# 5. Multi-Model Regression Benchmarking on Smooth Manifold Trajectory
cat("==============================================================================\n")
cat("BENCHMARKING MULTI-MODEL REGRESSIONS ON SMOOTH MANIFOLD\n")
cat("==============================================================================\n\n")

dep_params <- c("rho_base_mean", "tau_log_mean", "d_fb", "d_inh", "lambda_fb")
bench_list <- list()
coeff_list <- list()

calc_rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
calc_adj_r2 <- function(r2, n, p) 1 - (1 - r2) * (n - 1) / (n - p - 1)

for (dep in dep_params) {
  y_val <- ridge_df[[dep]]
  x_val <- ridge_df$d_in
  n_obs <- length(y_val)
  
  # 1. Linear Model
  lm_lin <- lm(y_val ~ x_val)
  sum_lin <- summary(lm_lin)
  r2_lin <- sum_lin$r.squared
  adj_r2_lin <- calc_adj_r2(r2_lin, n_obs, 1)
  rmse_lin <- calc_rmse(y_val, fitted(lm_lin))
  
  # 2. Quadratic Model
  lm_quad <- lm(y_val ~ poly(x_val, 2, raw = TRUE))
  sum_quad <- summary(lm_quad)
  c_quad <- coef(lm_quad)
  r2_quad <- sum_quad$r.squared
  adj_r2_quad <- calc_adj_r2(r2_quad, n_obs, 2)
  rmse_quad <- calc_rmse(y_val, fitted(lm_quad))
  
  # 3. Cubic Model
  lm_cub <- lm(y_val ~ poly(x_val, 3, raw = TRUE))
  sum_cub <- summary(lm_cub)
  c_cub <- coef(lm_cub)
  r2_cub <- sum_cub$r.squared
  adj_r2_cub <- calc_adj_r2(r2_cub, n_obs, 3)
  rmse_cub <- calc_rmse(y_val, fitted(lm_cub))
  
  # 4. Quartic Model
  lm_quart <- lm(y_val ~ poly(x_val, 4, raw = TRUE))
  sum_quart <- summary(lm_quart)
  c_quart <- coef(lm_quart)
  r2_quart <- sum_quart$r.squared
  adj_r2_quart <- calc_adj_r2(r2_quart, n_obs, 4)
  rmse_quart <- calc_rmse(y_val, fitted(lm_quart))
  
  # 5. Natural Spline Model (df = 4)
  lm_spline <- lm(y_val ~ ns(x_val, df = 4))
  sum_spline <- summary(lm_spline)
  r2_spline <- sum_spline$r.squared
  adj_r2_spline <- calc_adj_r2(r2_spline, n_obs, 4)
  rmse_spline <- calc_rmse(y_val, fitted(lm_spline))
  
  cat(sprintf("--- Dependent Parameter: %s ---\n", dep))
  cat(sprintf("  1. Linear:    R^2 = %.4f | Adj R^2 = %.4f | RMSE = %.6f\n", r2_lin, adj_r2_lin, rmse_lin))
  cat(sprintf("  2. Quadratic: R^2 = %.4f | Adj R^2 = %.4f | RMSE = %.6f\n", r2_quad, adj_r2_quad, rmse_quad))
  cat(sprintf("  3. Cubic:     R^2 = %.4f | Adj R^2 = %.4f | RMSE = %.6f\n", r2_cub, adj_r2_cub, rmse_cub))
  cat(sprintf("  4. Quartic:   R^2 = %.4f | Adj R^2 = %.4f | RMSE = %.6f\n", r2_quart, adj_r2_quart, rmse_quart))
  cat(sprintf("  5. NS Spline: R^2 = %.4f | Adj R^2 = %.4f | RMSE = %.6f\n\n", r2_spline, adj_r2_spline, rmse_spline))
  
  bench_list[[dep]] <- data.frame(
    Parameter = dep,
    Linear_R2 = r2_lin, Linear_RMSE = rmse_lin,
    Quad_R2 = r2_quad, Quad_RMSE = rmse_quad,
    Cubic_R2 = r2_cub, Cubic_RMSE = rmse_cub,
    Quartic_R2 = r2_quart, Quartic_RMSE = rmse_quart,
    Spline_R2 = r2_spline, Spline_RMSE = rmse_spline,
    stringsAsFactors = FALSE
  )
  
  coeff_list[[dep]] <- data.frame(
    Parameter = dep,
    Cubic_c0 = c_cub[1], Cubic_c1 = c_cub[2], Cubic_c2 = c_cub[3], Cubic_c3 = c_cub[4],
    Quartic_c0 = c_quart[1], Quartic_c1 = c_quart[2], Quartic_c2 = c_quart[3], Quartic_c3 = c_quart[4], Quartic_c4 = c_quart[5],
    stringsAsFactors = FALSE
  )
}

bench_df <- do.call(rbind, bench_list)
coeff_df <- do.call(rbind, coeff_list)
rownames(coeff_df) <- dep_params

write.csv(bench_df, "ridge_manifold_benchmarks.csv", row.names = FALSE)
write.csv(coeff_df, "ridge_manifold_coefficients.csv", row.names = FALSE)
cat("Saved multi-model benchmark metrics to ridge_manifold_benchmarks.csv\n")
cat("Saved algebraic coefficients to ridge_manifold_coefficients.csv\n\n")

# 6. Output Hardcodable C++ Code Block to Console
cat("==============================================================================\n")
cat("C++ HARDCODABLE SMOOTH RIDGE MANIFOLD EQUATIONS (Copy-Paste Ready)\n")
cat("==============================================================================\n")
cat("// Smooth Empirical Cerebellar Ridge Manifold Functions (Master Coordinate: d_in)\n")
cat("// Higher-Order Cubic Algebraic Polynomial Models\n")
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

# 7. Render Visualization Plot
p1 <- ggplot(ridge_df, aes(x = d_in, y = d_inh)) +
  geom_point(color = "black", alpha = 0.6, size = 2) +
  geom_smooth(aes(color = "Quadratic"), method = "lm", formula = y ~ poly(x, 2), se = FALSE, linewidth = 1) +
  geom_smooth(aes(color = "Cubic"), method = "lm", formula = y ~ poly(x, 3), se = FALSE, linewidth = 1) +
  geom_smooth(aes(color = "Spline (df=4)"), method = "lm", formula = y ~ ns(x, df = 4), se = FALSE, linetype = "dashed", linewidth = 1) +
  scale_color_manual(name = "Model", values = c("Quadratic" = "blue", "Cubic" = "red", "Spline (df=4)" = "purple4")) +
  labs(title = "Golgi Inhibition D(W_inh) vs D(W_in)", x = "D(W_in)", y = "D(W_inh)") +
  theme_minimal()

p2 <- ggplot(ridge_df, aes(x = d_in, y = tau_log_mean)) +
  geom_point(color = "black", alpha = 0.6, size = 2) +
  geom_smooth(aes(color = "Quadratic"), method = "lm", formula = y ~ poly(x, 2), se = FALSE, linewidth = 1) +
  geom_smooth(aes(color = "Cubic"), method = "lm", formula = y ~ poly(x, 3), se = FALSE, linewidth = 1) +
  geom_smooth(aes(color = "Spline (df=4)"), method = "lm", formula = y ~ ns(x, df = 4), se = FALSE, linetype = "dashed", linewidth = 1) +
  scale_color_manual(name = "Model", values = c("Quadratic" = "blue", "Cubic" = "red", "Spline (df=4)" = "purple4")) +
  labs(title = "Retention Tau_log_mean vs D(W_in)", x = "D(W_in)", y = "tau_log_mean") +
  theme_minimal()

p3 <- ggplot(ridge_df, aes(x = d_in, y = lambda_fb)) +
  geom_point(color = "black", alpha = 0.6, size = 2) +
  geom_smooth(aes(color = "Quadratic"), method = "lm", formula = y ~ poly(x, 2), se = FALSE, linewidth = 1) +
  geom_smooth(aes(color = "Cubic"), method = "lm", formula = y ~ poly(x, 3), se = FALSE, linewidth = 1) +
  geom_smooth(aes(color = "Spline (df=4)"), method = "lm", formula = y ~ ns(x, df = 4), se = FALSE, linetype = "dashed", linewidth = 1) +
  scale_color_manual(name = "Model", values = c("Quadratic" = "blue", "Cubic" = "red", "Spline (df=4)" = "purple4")) +
  labs(title = "Spectral Radius lambda_fb vs D(W_in)", x = "D(W_in)", y = "lambda_fb") +
  theme_minimal()

p4 <- ggplot(ridge_df, aes(x = d_in, y = Marginal_Fitness_J)) +
  geom_line(color = "darkgreen", linewidth = 1.2) +
  geom_point(color = "darkgreen", size = 2) +
  labs(title = "Marginalized Fitness (50 Steps)", x = "D(W_in)", y = "Marginal J") +
  theme_minimal()

g_ridge <- grid.arrange(p1, p2, p3, p4, ncol = 2)
ggsave("ridge_manifold_fits.png", g_ridge, width = 11, height = 9, dpi = 300)
ggsave("C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad/ridge_manifold_fits.png", g_ridge, width = 11, height = 9, dpi = 300)

cat("Rendered and saved smooth ridge manifold plot to ridge_manifold_fits.png\n")
cat("Extraction pipeline completed successfully.\n")
