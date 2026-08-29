# ==============================================================================
# EXACT-R: Pure R 12-Core Parallel GP Surrogate Modeling Pipeline
# 10D Joint Search Space Sampling, Black-Box Evaluation & GPR Fit
# Leaves 2 CPU cores free (12 parallel workers on 14-core machine)
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(Matrix)
  library(lhs)
  library(DiceKriging)
  library(ggplot2)
  library(jsonlite)
  library(parallel)
})

# Source External C++ & helper modules
source("data_generators.R")
source("evaluation_metrics.R")

cat("==============================================================================\n")
cat("Starting Pure R 10D GP Surrogate Modeling (12-Core Parallel Execution)\n")
cat("==============================================================================\n\n")

N_GC <- 1000          # Reservoir Granular Cell count
N_GoC <- 100          # Golgi Cell Bottleneck count
N_MF <- 100           # Mossy Fiber Input channels
N_actions <- 3        # 3AFC readout dimension
delta_t_const <- 0.01 # Timestep 10ms

# Dual-Space Linear Memory Capacity (Fast Linear Algebra)
compute_fast_mc <- function(Z_matrix, U_matrix, k_max = 15, ridge_lambda = 1e-3) {
  T_steps <- nrow(Z_matrix)
  N_channels <- ncol(U_matrix)
  if (T_steps <= k_max + 10) return(0.0)
  
  Z_centered <- scale(Z_matrix, center = TRUE, scale = FALSE)
  eval_channels <- min(5, N_channels)
  mc_total <- 0.0
  
  for (ch in 1:eval_channels) {
    u_channel <- U_matrix[, ch]
    mc_channel <- 0.0
    
    for (k in 1:k_max) {
      idx_z <- (k + 1):T_steps
      idx_u <- 1:(T_steps - k)
      
      Z_sub <- Z_centered[idx_z, , drop = FALSE]
      u_sub <- u_channel[idx_u]
      
      K_mat <- Z_sub %*% t(Z_sub)
      diag(K_mat) <- diag(K_mat) + ridge_lambda
      
      u_pred <- tryCatch({
        alpha_vec <- solve(K_mat, u_sub)
        u_sub - ridge_lambda * alpha_vec
      }, error = function(e) {
        rep(0, length(u_sub))
      })
      
      ss_tot <- sum((u_sub - mean(u_sub))^2)
      ss_res <- sum((u_sub - u_pred)^2)
      
      if (ss_tot > 1e-9) {
        r2_k <- max(0, 1 - (ss_res / ss_tot))
        mc_channel <- mc_channel + r2_k
      }
    }
    mc_total <- mc_total + mc_channel
  }
  return(mc_total / eval_channels)
}

# 10D LHS Generator
generate_lhs_10d <- function(n_samples = 1000, seed = 2026) {
  set.seed(seed)
  k <- 10
  raw_lhs <- lhs::randomLHS(n_samples, k)
  
  rho_base_mean <- 0.01 + raw_lhs[, 1] * (0.50 - 0.01)
  tau_log_mean  <- 0.50 + raw_lhs[, 2] * (3.00 - 0.50)
  d_in          <- 0.01 + raw_lhs[, 3] * (0.20 - 0.01)
  d_fb          <- 0.01 + raw_lhs[, 4] * (0.10 - 0.01)
  d_inh         <- 0.05 + raw_lhs[, 5] * (0.30 - 0.05)
  lambda_fb     <- 0.70 + raw_lhs[, 6] * (0.99 - 0.70)
  
  f_base        <- 0.10 + raw_lhs[, 7] * (5.00 - 0.10)
  A_load        <- 1.00 + raw_lhs[, 8] * (10.00 - 1.00)
  delta_phi     <- 0.00 + raw_lhs[, 9] * (pi - 0.00)
  sigma_noise   <- 0.01 + raw_lhs[, 10] * (1.00 - 0.01)
  
  data.frame(
    rho_base_mean = rho_base_mean,
    tau_log_mean  = tau_log_mean,
    d_in          = d_in,
    d_fb          = d_fb,
    d_inh         = d_inh,
    lambda_fb     = lambda_fb,
    f_base        = f_base,
    A_load        = A_load,
    delta_phi     = delta_phi,
    sigma_noise   = sigma_noise
  )
}

generate_phase_coupled_kinematic_data <- function(T_steps = 300, N_channels = 100, delta_t = 0.01,
                                                   f_base = 1.0, A_load = 1.0, delta_phi = 0.5 * pi, noise_sigma = 0.1) {
  time_vec <- (0:(T_steps - 1)) * delta_t
  n_basis <- 4
  basis_matrix <- matrix(0, nrow = T_steps, ncol = n_basis)
  
  for (b in 1:n_basis) {
    phi_b <- (b - 1) * (delta_phi / max(1, n_basis - 1))
    harm2 <- 0.5 * cos(4 * pi * f_base * time_vec + 2 * phi_b)
    basis_matrix[, b] <- A_load * (sin(2 * pi * f_base * time_vec + phi_b) + harm2)
  }
  
  set.seed(42 + round(f_base * 100) + round(A_load * 10))
  proj_matrix <- matrix(rnorm(n_basis * N_channels), nrow = n_basis, ncol = N_channels)
  u_raw <- basis_matrix %*% proj_matrix
  
  execution_noise <- matrix(rnorm(T_steps * N_channels, mean = 0, sd = noise_sigma), nrow = T_steps, ncol = N_channels)
  u_noisy <- u_raw + execution_noise
  1 / (1 + exp(-0.2 * u_noisy))
}

generate_topology <- function(d_in, d_fb, d_inh, d_collateral, lambda_fb) {
  W_in <- rsparsematrix(N_GC, N_MF, density = d_in)
  W_fb <- rsparsematrix(N_GoC, N_GC, density = d_fb)
  fb_norm <- max(1e-6, suppressWarnings(tryCatch(norm(as.matrix(W_fb), "2"), error = function(e) 1.0)))
  W_fb <- W_fb * (lambda_fb / fb_norm)
  W_inh <- rsparsematrix(N_GC, N_GoC, density = d_inh)
  W_collateral <- rsparsematrix(N_GoC, 1 + N_actions, density = d_collateral)
  list(W_in = W_in, W_fb = W_fb, W_inh = W_inh, W_collateral = W_collateral)
}

evaluate_sample <- function(sample_id, psi_row, T_pre = 300) {
  rho_mean  <- psi_row$rho_base_mean
  tau_mean  <- psi_row$tau_log_mean
  d_in      <- psi_row$d_in
  d_fb      <- psi_row$d_fb
  d_inh     <- psi_row$d_inh
  lambda_fb <- psi_row$lambda_fb
  
  f_base      <- psi_row$f_base
  A_load      <- psi_row$A_load
  delta_phi   <- psi_row$delta_phi
  sigma_noise <- psi_row$sigma_noise
  
  set.seed(sample_id + 10000)
  tau_vec <- rlnorm(N_GC, meanlog = tau_mean, sdlog = 0.50)
  tau_vec <- pmax(1.0, pmin(tau_vec, 1000.0))
  
  rho_base <- rnorm(N_GC, mean = rho_mean, sd = 0.05)
  rho_base <- pmax(0.001, pmin(rho_base, 0.95))
  
  topo <- generate_topology(d_in, d_fb, d_inh, 0.05, lambda_fb)
  
  model <- new(ExactRModel,
               topo$W_in, topo$W_fb, topo$W_inh, topo$W_collateral,
               rho_base, tau_vec, N_actions,
               0.05, 0.05, 1.5, 0.0)
  
  U_matrix <- generate_phase_coupled_kinematic_data(
    T_steps = T_pre, N_channels = N_MF, delta_t = delta_t_const,
    f_base = f_base, A_load = A_load, delta_phi = delta_phi, noise_sigma = sigma_noise
  )
  
  Z_matrix <- matrix(0, nrow = T_pre, ncol = N_GC)
  Omega_history <- numeric(T_pre)
  
  model$reset_state()
  for (t in 1:T_pre) {
    u_t <- U_matrix[t, ]
    fwd <- model$forward_pass(u_t, delta_t_const)
    Z_matrix[t, ] <- model$get_z_GC()
    bwd <- model$backward_pass(1, 0.0, 0.02)
    Omega_history[t] <- bwd$Omega_t
  }
  
  MC <- compute_fast_mc(Z_matrix, U_matrix, k_max = 15)
  kappa_rank <- compute_effective_kernel_rank(Z_matrix)
  
  model_pert <- new(ExactRModel,
                    topo$W_in, topo$W_fb, topo$W_inh, topo$W_collateral,
                    rho_base, tau_vec, N_actions,
                    0.05, 0.05, 1.5, 0.0)
  
  lambda_max <- compute_lyapunov_exponent(model, model_pert, U_matrix, delta_t = delta_t_const)
  var_omega <- compute_entropy_operational_range(Omega_history)
  alpha_rpe <- benchmark_rpe_acceleration(model, T_trials = 80, delta_t = delta_t_const)
  
  J_score <- compute_composite_fitness(MC, kappa_rank, lambda_max, var_omega, alpha_rpe)
  
  res <- psi_row
  res$Sample_ID  <- sample_id
  res$MC         <- MC
  res$kappa_rank <- kappa_rank
  res$lambda_max <- lambda_max
  res$var_omega  <- var_omega
  res$alpha_rpe  <- alpha_rpe
  res$Fitness_J  <- J_score
  return(res)
}

# Real-time progress updater
update_realtime_progress <- function(current, total, start_t, last_fitness, rolling_mean_j) {
  elapsed_s <- as.numeric(difftime(Sys.time(), start_t, units = "secs"))
  pct <- round((current / total) * 100, 1)
  rate <- current / max(1e-3, elapsed_s)
  eta_s <- round((total - current) / max(1e-3, rate))
  
  prog_obj <- list(
    current_sample = current,
    total_samples = total,
    pct_complete = pct,
    elapsed_seconds = round(elapsed_s, 1),
    eta_seconds = eta_s,
    samples_per_sec = round(rate, 2),
    last_fitness_J = round(last_fitness, 2),
    rolling_mean_fitness_J = round(rolling_mean_j, 2),
    status = ifelse(current == total, "COMPLETED", "RUNNING")
  )
  
  write(jsonlite::toJSON(prog_obj, auto_unbox = TRUE, pretty = TRUE), file = "gp_progress.json")
  
  txt_line <- sprintf("[%s] Progress: %d / %d (%.1f%%) | Rate: %.2f sps | Elapsed: %.1fs | ETA: %.1fs | Fitness J: %.2f",
                      format(Sys.time(), "%H:%M:%S"), current, total, pct, rate, elapsed_s, eta_s, last_fitness)
  writeLines(txt_line, con = "gp_progress.txt")
}

# ==============================================================================
# PHASE 1 & 2: 10D LHS Sampling & Parallel 12-Core Black-Box Evaluation
# ==============================================================================
N_samples <- 1000
cat(sprintf("Phase 1: Generating %d 10D Latin Hypercube Samples...\n", N_samples))
lhs_points <- generate_lhs_10d(n_samples = N_samples, seed = 2026)

# Setup 12-core parallel cluster (leaving 2 CPU cores free)
total_cores <- detectCores()
n_workers <- max(1, total_cores - 2)
cat(sprintf("Phase 2: Setting up %d-Core Parallel Worker Cluster (leaving 2 of %d cores free)...\n", n_workers, total_cores))

cl <- makeCluster(n_workers)

clusterEvalQ(cl, {
  suppressPackageStartupMessages({
    library(Rcpp)
    library(RcppEigen)
    library(Matrix)
  })
  source("data_generators.R")
  source("evaluation_metrics.R")
  sourceCpp("reservoir.cpp")
})

clusterExport(cl, c(
  "generate_phase_coupled_kinematic_data", "generate_topology",
  "evaluate_sample", "compute_fast_mc", "delta_t_const",
  "N_GC", "N_GoC", "N_MF", "N_actions"
))

cat("Starting Parallel Black-Box Evaluation Loop...\n\n")
batch_size <- n_workers
n_batches <- ceiling(N_samples / batch_size)
results_list <- vector("list", N_samples)
start_time <- Sys.time()

for (b in 1:n_batches) {
  start_idx <- (b - 1) * batch_size + 1
  end_idx <- min(N_samples, b * batch_size)
  chunk_ids <- start_idx:end_idx
  
  # Extract batch parameter rows
  batch_rows <- lapply(chunk_ids, function(idx) lhs_points[idx, ])
  
  # Evaluate batch in parallel across 12 workers
  batch_res <- parLapply(cl, 1:length(chunk_ids), function(k, ids, rows) {
    s_id <- ids[k]
    p_row <- rows[[k]]
    evaluate_sample(s_id, p_row, T_pre = 300)
  }, ids = chunk_ids, rows = batch_rows)
  
  for (k in 1:length(chunk_ids)) {
    results_list[[chunk_ids[k]]] <- batch_res[[k]]
  }
  
  # Real-time progress update
  last_fitness <- batch_res[[length(batch_res)]]$Fitness_J
  fitness_vals <- sapply(results_list[1:end_idx], function(x) x$Fitness_J)
  rolling_mean <- mean(fitness_vals)
  
  update_realtime_progress(end_idx, N_samples, start_time, last_fitness, rolling_mean)
  
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  rate <- end_idx / max(1e-3, elapsed)
  eta <- (N_samples - end_idx) / max(1e-3, rate)
  
  cat(sprintf("[PARALLEL 12-CORE %5.1f%%] Completed %4d / %d | Rate: %4.2f sps | Elapsed: %5.1fs | ETA: %5.1fs | Fitness J: %7.2f\n",
              (end_idx / N_samples) * 100, end_idx, N_samples, rate, elapsed, eta, last_fitness))
  flush.console()
}

stopCluster(cl)

full_df <- do.call(rbind, results_list)
write.csv(full_df, "gp_lhs_dataset.csv", row.names = FALSE)
cat("\nSaved 1000-point joint dataset to gp_lhs_dataset.csv\n\n")

# ==============================================================================
# PHASE 3: ARD Matérn 5/2 Gaussian Process Regression Fit (DiceKriging)
# ==============================================================================
cat("==============================================================================\n")
cat("Phase 3: Fitting ARD Matérn 5/2 Gaussian Process Model (DiceKriging)\n")
cat("==============================================================================\n")

param_cols <- c("rho_base_mean", "tau_log_mean", "d_in", "d_fb", "d_inh", "lambda_fb",
                "f_base", "A_load", "delta_phi", "sigma_noise")

X_raw <- as.matrix(full_df[, param_cols])
y_raw <- full_df$Fitness_J

bounds_lower <- c(0.01, 0.50, 0.01, 0.01, 0.05, 0.70, 0.10, 1.00, 0.00, 0.01)
bounds_upper <- c(0.50, 3.00, 0.20, 0.10, 0.30, 0.99, 5.00, 10.00, pi, 1.00)
ranges <- bounds_upper - bounds_lower

X_scaled <- t((t(X_raw) - bounds_lower) / ranges)
y_mean <- mean(y_raw)
y_sd <- sd(y_raw) + 1e-12
y_scaled <- (y_raw - y_mean) / y_sd

gp_model <- km(
  formula = ~1,
  design = X_scaled,
  response = y_scaled,
  covtype = "matern5_2",
  optim.method = "BFGS",
  control = list(trace = FALSE)
)

cat("DiceKriging GP Model Fitting Succeeded.\n")

l_scaled <- slot(gp_model@covariance, "range.val")
l_physical <- l_scaled * ranges
sensitivity_scores <- 1.0 / (l_scaled^2)

ard_ledger <- data.frame(
  Dimension = 1:10,
  Parameter = param_cols,
  Lengthscale_Scaled = l_scaled,
  Lengthscale_Physical = l_physical,
  Sensitivity_Score = sensitivity_scores,
  stringsAsFactors = FALSE
)
ard_ledger <- ard_ledger[order(-ard_ledger$Sensitivity_Score), ]
rownames(ard_ledger) <- NULL

cat("\n==============================================================================\n")
cat("THE ARD LENGTHSCALE LEDGER (Parameter Sensitivity Ranking)\n")
cat("==============================================================================\n")
print(ard_ledger)
write.csv(ard_ledger, "ard_lengthscale_ledger.csv", row.names = FALSE)

# ==============================================================================
# PHASE 4: Manifold Extraction & 2D Contour Plotting
# ==============================================================================
cat("\n==============================================================================\n")
cat("Phase 4: Manifold Extraction & Criticality Adaptation Solver\n")
cat("==============================================================================\n")

predict_gp <- function(psi_phys) {
  psi_scaled <- matrix((psi_phys - bounds_lower) / ranges, nrow = 1)
  pred <- predict(gp_model, newdata = psi_scaled, type = "UK")
  mu_phys <- pred$mean * y_sd + y_mean
  var_phys <- (pred$sd * y_sd)^2
  list(mean = mu_phys, var = var_phys)
}

neg_gp_obj <- function(theta_phys, phi_fixed) {
  psi_phys <- c(theta_phys, phi_fixed)
  -predict_gp(psi_phys)$mean
}

theta_lower <- bounds_lower[1:6]
theta_upper <- bounds_upper[1:6]
theta_init  <- 0.5 * (theta_lower + theta_upper)

phi_A <- c(0.2, 8.0, 0.5 * pi, 0.1)
opt_A <- optim(theta_init, neg_gp_obj, phi_fixed = phi_A, method = "L-BFGS-B", lower = theta_lower, upper = theta_upper)

phi_B <- c(3.0, 2.0, 0.5 * pi, 0.1)
opt_B <- optim(theta_init, neg_gp_obj, phi_fixed = phi_B, method = "L-BFGS-B", lower = theta_lower, upper = theta_upper)

cat("\nConditional Manifold Adaptation across Kinematic Regimes:\n")
cat(sprintf("Regime A (f_base = 0.2 Hz, A_load = 8.0): tau_log_mean = %.4f, lambda_fb = %.4f (Fitness J = %.2f)\n",
            opt_A$par[2], opt_A$par[6], -opt_A$value))
cat(sprintf("Regime B (f_base = 3.0 Hz, A_load = 2.0): tau_log_mean = %.4f, lambda_fb = %.4f (Fitness J = %.2f)\n",
            opt_B$par[2], opt_B$par[6], -opt_B$value))

f_seq <- seq(0.2, 3.0, length.out = 15)
tau_opt_seq <- numeric(length(f_seq))
lambda_opt_seq <- numeric(length(f_seq))

for (idx in seq_along(f_seq)) {
  f_val <- f_seq[idx]
  phi_f <- c(f_val, max(1.0, 5.0 - f_val), 0.5 * pi, 0.1)
  res_f <- optim(theta_init, neg_gp_obj, phi_fixed = phi_f, method = "L-BFGS-B", lower = theta_lower, upper = theta_upper)
  tau_opt_seq[idx] <- res_f$par[2]
  lambda_opt_seq[idx] <- res_f$par[6]
}

lm_tau <- lm(tau_opt_seq ~ f_seq)
lm_lambda <- lm(lambda_opt_seq ~ f_seq)

cat("\nClosed-Form Conditional Manifold Equations:\n")
cat(sprintf("  tau_log_mean(f_base)  = %+.4f * f_base + %.4f\n", coef(lm_tau)[2], coef(lm_tau)[1]))
cat(sprintf("  lambda_fb(f_base)     = %+.4f * f_base + %.4f\n", coef(lm_lambda)[2], coef(lm_lambda)[1]))

manifold_summary <- data.frame(
  f_base = f_seq,
  tau_log_mean_opt = tau_opt_seq,
  lambda_fb_opt = lambda_opt_seq
)
write.csv(manifold_summary, "conditional_manifold_results.csv", row.names = FALSE)

# Render 2D Contour Plots
top_2_names <- ard_ledger$Parameter[1:2]
idx1 <- which(param_cols == top_2_names[1])
idx2 <- which(param_cols == top_2_names[2])

cat(sprintf("\nRendering 2D Contour Plot over Top 2 Sensitive Parameters: %s vs %s...\n",
            top_2_names[1], top_2_names[2]))

grid_res <- 60
p1_seq <- seq(bounds_lower[idx1], bounds_upper[idx1], length.out = grid_res)
p2_seq <- seq(bounds_lower[idx2], bounds_upper[idx2], length.out = grid_res)

grid_df <- expand.grid(P1 = p1_seq, P2 = p2_seq)
mu_vals <- numeric(nrow(grid_df))
var_vals <- numeric(nrow(grid_df))

nominal_psi <- 0.5 * (bounds_lower + bounds_upper)
nominal_psi[1:6] <- opt_A$par

for (r in 1:nrow(grid_df)) {
  eval_psi <- nominal_psi
  eval_psi[idx1] <- grid_df$P1[r]
  eval_psi[idx2] <- grid_df$P2[r]
  pred <- predict_gp(eval_psi)
  mu_vals[r] <- pred$mean
  var_vals[r] <- pred$var
}

grid_df$Mean <- mu_vals
grid_df$Variance <- var_vals

p1 <- ggplot(grid_df, aes(x = P1, y = P2, z = Mean)) +
  geom_contour_filled(bins = 20) +
  scale_fill_viridis_d(option = "C", name = "Mean J") +
  labs(title = paste("GP Predictive Mean mu_*:", top_2_names[1], "vs", top_2_names[2]),
       x = top_2_names[1], y = top_2_names[2]) +
  theme_minimal(base_size = 12)

p2 <- ggplot(grid_df, aes(x = P1, y = P2, z = Variance)) +
  geom_contour_filled(bins = 20) +
  scale_fill_viridis_d(option = "B", name = "Variance") +
  labs(title = paste("GP Predictive Variance sigma_*^2:", top_2_names[1], "vs", top_2_names[2]),
       x = top_2_names[1], y = top_2_names[2]) +
  theme_minimal(base_size = 12)

suppressPackageStartupMessages(library(gridExtra))
g_contour <- grid.arrange(p1, p2, ncol = 2)

ggsave("gp_surrogate_contours.png", g_contour, width = 12, height = 5.5, dpi = 300)
ggsave("C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad/gp_surrogate_contours.png", g_contour, width = 12, height = 5.5, dpi = 300)

cat("\n==============================================================================\n")
cat("PURE R 12-CORE PARALLEL GP SURROGATE PIPELINE COMPLETED SUCCESSFULLY\n")
cat("==============================================================================\n")
