# ==============================================================================
# EXACT-R: Gaussian Process Surrogate Modeling Pipeline (Phase 1 & 2)
# 10D Joint Search Space Sampling & Black-Box Evaluation
# ==============================================================================
library(Rcpp)
library(RcppEigen)
library(Matrix)

# 1. Source External Modules
source("data_generators.R")
source("evaluation_metrics.R")
sourceCpp("reservoir.cpp")

cat("==============================================================================\n")
cat("Starting 10D Joint Space Sampling & Evaluation for GP Surrogate Modeling\n")
cat("==============================================================================\n")

# Setup Dimensions
N_GC <- 1000          # Granular Cells
N_GoC <- 100          # Golgi Bottleneck
N_MF <- 100           # Mossy Fiber Channels
N_actions <- 3        # 3AFC readout
delta_t_const <- 0.01 # 10ms timestep

# Latin Hypercube Sampling generator
generate_lhs_10d <- function(n_samples = 1000, seed = 2026) {
  set.seed(seed)
  k <- 10
  
  if (requireNamespace("lhs", quietly = TRUE)) {
    raw_lhs <- lhs::randomLHS(n_samples, k)
  } else {
    raw_lhs <- matrix(0, nrow = n_samples, ncol = k)
    for (j in 1:k) {
      idx <- sample(1:n_samples)
      raw_lhs[, j] <- (idx - runif(n_samples)) / n_samples
    }
  }
  
  # Map 10D hypercube [0, 1]^10 to exact parameter ranges
  # Subspace 1: Structural Parameters (\Theta)
  rho_base_mean <- 0.01 + raw_lhs[, 1] * (0.50 - 0.01)
  tau_log_mean  <- 0.50 + raw_lhs[, 2] * (3.00 - 0.50)
  d_in          <- 0.01 + raw_lhs[, 3] * (0.20 - 0.01)
  d_fb          <- 0.01 + raw_lhs[, 4] * (0.10 - 0.01)
  d_inh         <- 0.05 + raw_lhs[, 5] * (0.30 - 0.05)
  lambda_fb     <- 0.70 + raw_lhs[, 6] * (0.99 - 0.70)
  
  # Subspace 2: Generative Kinematic Parameters (\Phi)
  f_base        <- 0.10 + raw_lhs[, 7] * (5.00 - 0.10)
  A_load        <- 1.00 + raw_lhs[, 8] * (10.00 - 1.00)
  delta_phi     <- 0.00 + raw_lhs[, 9] * (pi - 0.00)
  sigma_noise   <- 0.01 + raw_lhs[, 10] * (1.00 - 0.01)
  
  psi_df <- data.frame(
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
  return(psi_df)
}

# Phase-Coupled Kinematic Data Generator
generate_phase_coupled_kinematic_data <- function(T_steps = 400, N_channels = 100, delta_t = 0.01,
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
  
  u_data <- 1 / (1 + exp(-0.2 * u_noisy))
  return(u_data)
}

# Network Topology Helper
generate_topology <- function(d_in, d_fb, d_inh, d_collateral, lambda_fb) {
  W_in <- rsparsematrix(N_GC, N_MF, density = d_in)
  W_fb <- rsparsematrix(N_GoC, N_GC, density = d_fb)
  
  fb_norm <- max(1e-6, suppressWarnings(tryCatch(norm(as.matrix(W_fb), "2"), error = function(e) 1.0)))
  W_fb <- W_fb * (lambda_fb / fb_norm)
  
  W_inh <- rsparsematrix(N_GC, N_GoC, density = d_inh)
  W_collateral <- rsparsematrix(N_GoC, 1 + N_actions, density = d_collateral)
  
  return(list(W_in = W_in, W_fb = W_fb, W_inh = W_inh, W_collateral = W_collateral))
}

# Single trial evaluator
evaluate_joint_sample <- function(sample_id, psi_row, T_pre = 400) {
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
  
  rho_sd <- 0.05
  tau_sd <- 0.50
  d_collateral <- 0.05
  
  set.seed(sample_id + 10000)
  tau_vec <- rlnorm(N_GC, meanlog = tau_mean, sdlog = tau_sd)
  tau_vec <- pmax(1.0, pmin(tau_vec, 1000.0))
  
  rho_base <- rnorm(N_GC, mean = rho_mean, sd = rho_sd)
  rho_base <- pmax(0.001, pmin(rho_base, 0.95))
  
  topo <- generate_topology(d_in, d_fb, d_inh, d_collateral, lambda_fb)
  
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
  
  MC <- compute_linear_memory_capacity(Z_matrix, U_matrix, k_max = 15)
  kappa_rank <- compute_effective_kernel_rank(Z_matrix)
  
  model_pert <- new(ExactRModel,
                    topo$W_in, topo$W_fb, topo$W_inh, topo$W_collateral,
                    rho_base, tau_vec, N_actions,
                    0.05, 0.05, 1.5, 0.0)
  
  lambda_max <- compute_lyapunov_exponent(model, model_pert, U_matrix, delta_t = delta_t_const)
  var_omega <- compute_entropy_operational_range(Omega_history)
  alpha_rpe <- benchmark_rpe_acceleration(model, T_trials = 150, delta_t = delta_t_const)
  
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

# Main Execution
N_samples <- 1000
cat(sprintf("Generating %d LHS points across 10D search space...\n", N_samples))
lhs_points <- generate_lhs_10d(n_samples = N_samples, seed = 2026)

cat("Starting black-box evaluation loop...\n")
results_list <- vector("list", N_samples)
start_time <- Sys.time()

for (i in 1:N_samples) {
  res <- evaluate_joint_sample(i, lhs_points[i, ], T_pre = 400)
  results_list[[i]] <- res
  
  if (i %% 100 == 0 || i == N_samples) {
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    cat(sprintf("Evaluated %4d / %d samples | Elapsed: %.1fs | Last Fitness J: %7.2f\n",
                i, N_samples, elapsed, res$Fitness_J))
  }
}

full_df <- do.call(rbind, results_list)
write.csv(full_df, "gp_lhs_dataset.csv", row.names = FALSE)
cat("\nSaved 1000-point joint dataset to gp_lhs_dataset.csv\n")
cat("Phase 1 & 2 completed successfully.\n")
