# ==============================================================================
# EXACT-R: System Identification & Pre-Training Optimization Pipeline
# ==============================================================================
library(Rcpp)
library(RcppEigen)
library(Matrix)
library(ggplot2)
library(dplyr)

# Set working directory to project folder
if (requireNamespace("this.path", quietly = TRUE)) {
  setwd(this.path::here())
}

# 1. Source External Modules
source("data_generators.R")
source("evaluation_metrics.R")
sourceCpp("reservoir.cpp")

cat("==============================================================================\n")
cat("Starting System Identification & Pre-Training Optimization for ExactRModel\n")
cat("==============================================================================\n")

# Dimensions & Environment Setup
N_GC <- 1000          # Reservoir Granular Cell count
N_GoC <- 100          # Golgi Cell Bottleneck count
N_MF <- 100           # Mossy Fiber Input channels
N_actions <- 3        # 3AFC readout dimension
delta_t_const <- 0.01 # Timestep 10ms

# Topologies Generator Helper
generate_topology <- function(d_in, d_fb, d_inh, d_collateral, lambda_fb) {
  # W_in: N_GC x N_MF
  W_in <- rsparsematrix(N_GC, N_MF, density = d_in)
  
  # W_fb: N_GoC x N_GC
  W_fb <- rsparsematrix(N_GoC, N_GC, density = d_fb)
  
  # Spectral radius scaling for W_fb
  # Scale matrix norm / max eigenvalue approximation to lambda_fb
  fb_norm <- max(1e-6, suppressWarnings(tryCatch(norm(as.matrix(W_fb), "2"), error = function(e) 1.0)))
  W_fb <- W_fb * (lambda_fb / fb_norm)
  
  # W_inh: N_GC x N_GoC
  W_inh <- rsparsematrix(N_GC, N_GoC, density = d_inh)
  
  # W_collateral: N_GoC x (1 + N_actions)
  W_collateral <- rsparsematrix(N_GoC, 1 + N_actions, density = d_collateral)
  
  return(list(W_in = W_in, W_fb = W_fb, W_inh = W_inh, W_collateral = W_collateral))
}

# Function to run a single trial evaluation
evaluate_trial <- function(trial_id, protocol_name, rho_base_mean, rho_base_sd,
                           tau_log_mean, tau_log_sd, d_in, d_fb, d_inh, d_collateral,
                           lambda_fb, T_pre = 500) {
  
  # 1. Sample retention & time constant vectors
  tau_vec <- rlnorm(N_GC, meanlog = tau_log_mean, sdlog = tau_log_sd)
  tau_vec <- pmax(1.0, pmin(tau_vec, 1000.0))
  
  rho_base <- rnorm(N_GC, mean = rho_base_mean, sd = rho_base_sd)
  rho_base <- pmax(0.001, pmin(rho_base, 0.95))
  
  # 2. Build topology matrices
  topo <- generate_topology(d_in, d_fb, d_inh, d_collateral, lambda_fb)
  
  # 3. Instantiate model
  model <- new(ExactRModel,
               topo$W_in, topo$W_fb, topo$W_inh, topo$W_collateral,
               rho_base, tau_vec, N_actions,
               0.05, 0.05, 1.5, 0.0) # lr_v, lr_pi, k_ent, b_v_init
  
  # 4. Generate Pre-Training Dataset U \in R^{T \times N_MF}
  U_matrix <- generate_pretraining_data(protocol_name, T_steps = T_pre, N_channels = N_MF, delta_t = delta_t_const)
  
  # 5. Collect Reservoir States Z \in R^{T \times N_GC} and Gating Omega_t
  Z_matrix <- matrix(0, nrow = T_pre, ncol = N_GC)
  Omega_history <- numeric(T_pre)
  
  model$reset_state()
  for (t in 1:T_pre) {
    u_t <- U_matrix[t, ]
    fwd <- model$forward_pass(u_t, delta_t_const)
    Z_matrix[t, ] <- model$get_z_GC()
    
    # Run dummy backward pass to query Omega_t gate
    bwd <- model$backward_pass(1, 0.0, 0.02)
    Omega_history[t] <- bwd$Omega_t
  }
  
  # 6. Compute Evaluation Metrics
  MC <- compute_linear_memory_capacity(Z_matrix, U_matrix, k_max = 15)
  kappa_rank <- compute_effective_kernel_rank(Z_matrix)
  
  # Dual model for Lyapunov Exponent calculation
  model_pert <- new(ExactRModel,
                    topo$W_in, topo$W_fb, topo$W_inh, topo$W_collateral,
                    rho_base, tau_vec, N_actions,
                    0.05, 0.05, 1.5, 0.0)
  
  lambda_max <- compute_lyapunov_exponent(model, model_pert, U_matrix, delta_t = delta_t_const)
  var_omega <- compute_entropy_operational_range(Omega_history)
  alpha_rpe <- benchmark_rpe_acceleration(model, T_trials = 150, delta_t = delta_t_const)
  
  # Composite Fitness Score J
  J_score <- compute_composite_fitness(MC, kappa_rank, lambda_max, var_omega, alpha_rpe)
  
  return(data.frame(
    Trial = trial_id,
    Protocol = protocol_name,
    rho_base_mean = rho_base_mean,
    rho_base_sd = rho_base_sd,
    tau_log_mean = tau_log_mean,
    tau_log_sd = tau_log_sd,
    d_in = d_in,
    d_fb = d_fb,
    d_inh = d_inh,
    d_collateral = d_collateral,
    lambda_fb = lambda_fb,
    MC = MC,
    kappa_rank = kappa_rank,
    lambda_max = lambda_max,
    var_omega = var_omega,
    alpha_rpe = alpha_rpe,
    Fitness_J = J_score,
    stringsAsFactors = FALSE
  ))
}

# ==============================================================================
# PHASE 1: Exploration Sweep over 100 Trials
# ==============================================================================
cat("\n--- PHASE 1: Executing 100-Trial Exploration Sweep ---\n")
set.seed(42)

protocols <- c("Poisson", "Filtered", "Lorenz", "Kinematic")
N_trials <- 100
trial_results <- list()

for (i in 1:N_trials) {
  # Round-robin sampling to guarantee exactly 25 trials per protocol (100 total)
  prot <- protocols[((i - 1) %% length(protocols)) + 1]
  
  rho_mean <- runif(1, 0.01, 0.50)
  rho_sd <- runif(1, 0.001, 0.10)
  tau_mean <- runif(1, 0.5, 3.0)
  tau_sd <- runif(1, 0.1, 1.0)
  
  d_in <- runif(1, 0.01, 0.20)
  d_fb <- runif(1, 0.01, 0.10)
  d_inh <- runif(1, 0.05, 0.30)
  d_col <- runif(1, 0.01, 0.10)
  lambda_fb <- runif(1, 0.70, 0.99)
  
  res <- evaluate_trial(i, prot, rho_mean, rho_sd, tau_mean, tau_sd,
                        d_in, d_fb, d_inh, d_col, lambda_fb, T_pre = 400)
  
  trial_results[[i]] <- res
  
  if (i %% 10 == 0 || i == N_trials) {
    cat(sprintf("Trial %3d / %d | Protocol: %-9s | MC: %6.2f | Rank: %6.2f | Lambda: %6.4f | Fitness J: %7.2f\n",
                i, N_trials, res$Protocol, res$MC, res$kappa_rank, res$lambda_max, res$Fitness_J))
  }
}

trials_df <- do.call(rbind, trial_results)
write.csv(trials_df, "optimization_sweep_results.csv", row.names = FALSE)
cat("Exploration phase results logged to optimization_sweep_results.csv\n")

# ==============================================================================
# PHASE 2: Exploitation & Sensitivity Analysis
# ==============================================================================
cat("\n--- PHASE 2: Exploitation Phase & Sensitivity Analysis ---\n")

# Select top 5 Pareto candidate configurations by Fitness_J
top5_candidates <- trials_df %>%
  arrange(desc(Fitness_J)) %>%
  head(5)

cat("\nTop 5 Pareto-Optimal Candidate Initializations:\n")
print(top5_candidates[, c("Trial", "Protocol", "MC", "kappa_rank", "lambda_max", "var_omega", "alpha_rpe", "Fitness_J")])

# Perform sensitivity grid analysis on W_fb density vs W_inh density around top candidate
best_cand <- top5_candidates[1, ]
cat(sprintf("\nBest Candidate Protocol: %s (Trial %d, Fitness J: %.2f)\n", best_cand$Protocol, best_cand$Trial, best_cand$Fitness_J))

cat("Performing 2D Sensitivity Analysis on D(W_fb) vs D(W_inh)...\n")
d_fb_grid <- seq(0.01, 0.10, length.out = 5)
d_inh_grid <- seq(0.05, 0.30, length.out = 5)
sens_results <- list()
idx <- 1

for (fb_d in d_fb_grid) {
  for (inh_d in d_inh_grid) {
    res_sens <- evaluate_trial(
      trial_id = 1000 + idx,
      protocol_name = best_cand$Protocol,
      rho_base_mean = best_cand$rho_base_mean,
      rho_base_sd = best_cand$rho_base_sd,
      tau_log_mean = best_cand$tau_log_mean,
      tau_log_sd = best_cand$tau_log_sd,
      d_in = best_cand$d_in,
      d_fb = fb_d,
      d_inh = inh_d,
      d_collateral = best_cand$d_collateral,
      lambda_fb = best_cand$lambda_fb,
      T_pre = 400
    )
    sens_results[[idx]] <- res_sens
    idx <- idx + 1
  }
}
sens_df <- do.call(rbind, sens_results)
write.csv(sens_df, "sensitivity_analysis_results.csv", row.names = FALSE)

# ==============================================================================
# PHASE 3: Protocol Performance Summary & Final Parameters Output
# ==============================================================================
cat("\n==============================================================================\n")
cat("PERFORMANCE SUMMARY TABLE ACROSS PRE-TRAINING DATA PROTOCOLS\n")
cat("==============================================================================\n")

protocol_summary <- trials_df %>%
  group_by(Protocol) %>%
  summarize(
    Count = n(),
    Mean_MC = mean(MC),
    Max_MC = max(MC),
    Mean_Rank = mean(kappa_rank),
    Mean_Lambda_Max = mean(lambda_max),
    Mean_Var_Omega = mean(var_omega),
    Mean_Alpha_RPE = mean(alpha_rpe),
    Mean_Fitness = mean(Fitness_J),
    Max_Fitness = max(Fitness_J),
    .groups = "drop"
  ) %>%
  arrange(desc(Max_Fitness))

print(protocol_summary)

best_overall <- trials_df %>% arrange(desc(Fitness_J)) %>% slice(1)

cat("\n==============================================================================\n")
cat("OPTIMAL PRE-TRAINED INITIALIZATION PARAMETERS\n")
cat("==============================================================================\n")
cat(sprintf("Optimal Synthetic Input Generator (D_best): %s\n", best_overall$Protocol))
cat(sprintf("Recommended Exposure Duration (T_pre)     : 1000 timesteps\n"))
cat(sprintf("rho_base Distribution                     : Normal(mean = %.4f, sd = %.4f)\n", best_overall$rho_base_mean, best_overall$rho_base_sd))
cat(sprintf("tau_vector Distribution                   : LogNormal(meanlog = %.4f, sdlog = %.4f)\n", best_overall$tau_log_mean, best_overall$tau_log_sd))
cat(sprintf("Optimal Spectral Radius Scaling (lambda_fb): %.4f\n", best_overall$lambda_fb))
cat(sprintf("Connectivity Density D(W_in)              : %.4f\n", best_overall$d_in))
cat(sprintf("Connectivity Density D(W_fb)              : %.4f\n", best_overall$d_fb))
cat(sprintf("Connectivity Density D(W_inh)             : %.4f\n", best_overall$d_inh))
cat(sprintf("Connectivity Density D(W_collateral)      : %.4f\n", best_overall$d_collateral))
cat(sprintf("Achieved Fitness Score J                   : %.2f\n", best_overall$Fitness_J))
cat("==============================================================================\n")
