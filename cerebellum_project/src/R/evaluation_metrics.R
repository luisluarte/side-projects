# ==============================================================================
# EXACT-R: Evaluation Metrics & Multi-Objective Fitness Function (\mathcal{J})
# ==============================================================================

# 1. Linear Memory Capacity (MC)
compute_linear_memory_capacity <- function(Z_matrix, U_matrix, k_max = 20, ridge_lambda = 1e-3) {
  T_steps <- nrow(Z_matrix)
  N_channels <- ncol(U_matrix)
  
  if (T_steps <= k_max + 10) return(0.0)
  
  # Normalize Z_matrix for numerical stability
  Z_centered <- scale(Z_matrix, center = TRUE, scale = FALSE)
  
  mc_total <- 0.0
  # Average MC across representative input channels (e.g. up to 5 channels for speed)
  eval_channels <- min(5, N_channels)
  
  for (ch in 1:eval_channels) {
    u_channel <- U_matrix[, ch]
    mc_channel <- 0.0
    
    for (k in 1:k_max) {
      # Shift input back by k timesteps
      idx_z <- (k + 1):T_steps
      idx_u <- 1:(T_steps - k)
      
      Z_sub <- Z_centered[idx_z, , drop = FALSE]
      u_sub <- u_channel[idx_u]
      
      # Ridge regression fit
      # W_k = (Z^T Z + \lambda I)^(-1) Z^T u
      XtX <- t(Z_sub) %*% Z_sub
      diag(XtX) <- diag(XtX) + ridge_lambda
      Xty <- t(Z_sub) %*% u_sub
      
      w_k <- tryCatch({
        solve(XtX, Xty)
      }, error = function(e) {
        rep(0, ncol(Z_sub))
      })
      
      u_pred <- Z_sub %*% w_k
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

# 2. Effective Kernel Rank (\kappa_{rank})
compute_effective_kernel_rank <- function(Z_matrix) {
  T_steps <- nrow(Z_matrix)
  N_GC <- ncol(Z_matrix)
  
  # Covariance matrix C = 1/T Z^T Z
  Z_centered <- scale(Z_matrix, center = TRUE, scale = FALSE)
  C_mat <- (t(Z_centered) %*% Z_centered) / T_steps
  
  # Singular value decomposition
  svd_res <- tryCatch({
    svd(C_mat, nu = 0, nv = 0)$d
  }, error = function(e) {
    rep(0, N_GC)
  })
  
  svd_sum <- sum(svd_res) + 1e-12
  sigma_bar <- svd_res / svd_sum
  
  # Spectral entropy
  entropy <- 0.0
  for (s in sigma_bar) {
    if (s > 1e-12) {
      entropy <- entropy - s * log(s)
    }
  }
  
  kappa_rank <- exp(entropy)
  return(kappa_rank)
}

# 3. Maximum Lyapunov Exponent (\lambda_{max})
compute_lyapunov_exponent <- function(model_base, model_pert, U_matrix, delta_t = 0.01, pert_mag = 1e-6) {
  T_steps <- nrow(U_matrix)
  
  # Base run initial state
  model_base$reset_state()
  model_pert$reset_state()
  
  # Perturb initial state of model_pert
  z_init <- model_base$get_z_GC_prev()
  pert_vec <- rnorm(length(z_init))
  pert_vec <- pert_mag * (pert_vec / sqrt(sum(pert_vec^2)))
  model_pert$set_z_GC_prev(z_init + pert_vec)
  
  # Drive both models with identical input
  z_diff_final <- pert_mag
  for (t in 1:T_steps) {
    u_t <- U_matrix[t, ]
    fwd_base <- model_base$forward_pass(u_t, delta_t)
    fwd_pert <- model_pert$forward_pass(u_t, delta_t)
    
    z_base <- model_base$get_z_GC()
    z_pert <- model_pert$get_z_GC()
    
    if (t == T_steps) {
      z_diff_final <- sqrt(sum((z_base - z_pert)^2))
    }
  }
  
  lambda_max <- (1.0 / (T_steps * delta_t)) * log(max(1e-12, z_diff_final) / pert_mag)
  return(lambda_max)
}

# 4. Information Entropy Operational Range Var(\Omega_t)
compute_entropy_operational_range <- function(Omega_history) {
  if (length(Omega_history) < 2) return(0.0)
  return(var(Omega_history))
}

# 5. Downstream RPE Acceleration Factor (\alpha_{RPE}) on 3AFC Task
benchmark_rpe_acceleration <- function(model, T_trials = 300, delta_t = 0.01, ttf_t = 0.02) {
  model$reset_state()
  
  # 3-Alternative Forced Choice task
  # Action 1 gives reward 1.0, actions 2 and 3 give 0.0
  target_action <- 1
  N_channels <- 100
  
  rpe_history <- numeric(T_trials)
  
  for (t in 1:T_trials) {
    # Motor/sensory cue input
    u_t <- rep(0.1, N_channels)
    u_t[1:10] <- 0.8
    
    fwd <- model$forward_pass(u_t, delta_t)
    pi_t <- fwd$Policy
    
    # Action selection (1-indexed for 3AFC: 1, 2, 3)
    a_t <- sample(1:length(pi_t), size = 1, prob = pi_t)
    r_t <- ifelse(a_t == target_action, 1.0, 0.0)
    
    bwd <- model$backward_pass(a_t, r_t, ttf_t)
    rpe_history[t] <- abs(bwd$RPE)
  }
  
  # Measure mean RPE magnitude in late trials (second half)
  late_trials <- rpe_history[(floor(T_trials / 2) + 1):T_trials]
  mean_late_rpe <- mean(late_trials)
  
  # Acceleration factor: inverse of late RPE error
  alpha_rpe <- 1.0 / (mean_late_rpe + 1e-3)
  return(alpha_rpe)
}

# 6. Composite Fitness Objective J(\Theta, \mathcal{D})
compute_composite_fitness <- function(MC, kappa_rank, lambda_max, var_omega, alpha_rpe,
                                       w1 = 1.0, w2 = 0.05, w3 = 20.0, w4 = 10.0, w5 = 1.0) {
  # Constraint on \lambda_{max} target 0.00
  lambda_penalty <- abs(lambda_max - 0.00)
  
  J <- (w1 * MC) + (w2 * kappa_rank) - (w3 * lambda_penalty) + (w4 * var_omega) + (w5 * alpha_rpe)
  return(J)
}
