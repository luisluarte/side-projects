# ==============================================================================
# EXACT-R: Optimal Short-Horizon Time-Series Cross-Generalization Benchmark (4x4)
# ==============================================================================
library(Rcpp)
library(RcppEigen)
library(Matrix)
library(dplyr)

# Source required modules
source("data_generators.R")
source("evaluation_metrics.R")
sourceCpp("reservoir.cpp")

# Helper: Short-Horizon (k=1..3) Rolling-Origin Time-Series Cross-Validation
evaluate_shorthorizon_cv_prediction <- function(model, U_matrix, k_max = 3, n_folds = 5, delta_t = 0.01, n_components = 25) {
  T_steps <- nrow(U_matrix)
  N_channels <- ncol(U_matrix)
  N_GC <- 1000
  
  # Collect full reservoir state matrix Z \in R^{T x N_GC}
  Z_matrix <- matrix(0, nrow = T_steps, ncol = N_GC)
  model$reset_state()
  
  for (t in 1:T_steps) {
    u_t <- U_matrix[t, ]
    fwd <- model$forward_pass(u_t, delta_t)
    Z_matrix[t, ] <- model$get_z_GC()
  }
  
  # PCA dimension reduction to top n_components
  Z_centered <- scale(Z_matrix, center = TRUE, scale = FALSE)
  svd_z <- tryCatch(svd(Z_centered, nu = min(n_components, T_steps, N_GC), nv = 0), error = function(e) NULL)
  
  if (is.null(svd_z) || is.null(svd_z$u)) {
    set.seed(42)
    proj <- matrix(rnorm(N_GC * n_components), nrow = N_GC, ncol = n_components)
    Z_proj <- Z_centered %*% proj
  } else {
    n_comp_actual <- min(n_components, ncol(svd_z$u))
    Z_proj <- svd_z$u[, 1:n_comp_actual, drop = FALSE] %*% diag(svd_z$d[1:n_comp_actual], nrow = n_comp_actual)
  }
  
  # Add intercept
  Z_proj <- cbind(1, Z_proj)
  
  block_size <- floor(T_steps / n_folds)
  fold_scores <- numeric(n_folds - 1)
  eval_channels <- min(5, N_channels)
  
  for (f in 1:(n_folds - 1)) {
    train_idx <- 1:(f * block_size)
    test_idx <- ((f * block_size) + 1):min(T_steps, (f + 1) * block_size)
    
    Z_tr_raw <- Z_proj[train_idx, , drop = FALSE]
    Z_te_raw <- Z_proj[test_idx, , drop = FALSE]
    
    fold_r2_sum <- 0.0
    valid_k_count <- 0
    
    for (ch in 1:eval_channels) {
      u_train <- U_matrix[train_idx, ch]
      u_test <- U_matrix[test_idx, ch]
      
      for (k in 1:k_max) {
        if (length(u_train) <= k + 5 || length(u_test) <= k + 5) next
        
        Z_tr_sub <- Z_tr_raw[(k + 1):nrow(Z_tr_raw), , drop = FALSE]
        u_tr_sub <- u_train[1:(length(u_train) - k)]
        
        XtX <- t(Z_tr_sub) %*% Z_tr_sub
        diag(XtX) <- diag(XtX) + 1e-2
        Xty <- t(Z_tr_sub) %*% u_tr_sub
        
        w_k <- tryCatch(solve(XtX, Xty), error = function(e) rep(0, ncol(Z_tr_sub)))
        
        Z_te_sub <- Z_te_raw[(k + 1):nrow(Z_te_raw), , drop = FALSE]
        u_te_sub <- u_test[1:(length(u_test) - k)]
        
        u_pred <- Z_te_sub %*% w_k
        ss_tot <- sum((u_te_sub - mean(u_te_sub))^2)
        ss_res <- sum((u_te_sub - u_pred)^2)
        
        if (ss_tot > 1e-8) {
          r2_val <- max(0, min(1.0, 1 - (ss_res / ss_tot)))
          fold_r2_sum <- fold_r2_sum + r2_val
          valid_k_count <- valid_k_count + 1
        }
      }
    }
    fold_scores[f] <- ifelse(valid_k_count > 0, fold_r2_sum / valid_k_count, 0.0)
  }
  
  return(mean(fold_scores))
}

# Main function to run the 4x4 Cross-Generalization Matrix Benchmark
run_cross_generalization_benchmark <- function(sweep_results_file = "optimization_sweep_results.csv", T_test = 800) {
  cat("\n==============================================================================\n")
  cat("RUNNING SHORT-HORIZON TIME-SERIES CROSS-GENERALIZATION BENCHMARK (4x4 MATRIX)\n")
  cat("==============================================================================\n")
  
  protocols <- c("Kinematic", "Filtered", "Lorenz", "Poisson")
  N_GC <- 1000
  N_GoC <- 100
  N_MF <- 100
  N_actions <- 3
  
  trials_df <- read.csv(sweep_results_file, stringsAsFactors = FALSE)
  
  parent_models <- list()
  for (prot in protocols) {
    top_cand <- trials_df %>%
      filter(Protocol == prot) %>%
      arrange(desc(Fitness_J)) %>%
      slice(1)
    
    if (nrow(top_cand) == 0) {
      top_cand <- trials_df %>% arrange(desc(Fitness_J)) %>% slice(1)
    }
    
    set.seed(100 + which(protocols == prot))
    tau_vec <- rlnorm(N_GC, meanlog = top_cand$tau_log_mean, sdlog = top_cand$tau_log_sd)
    tau_vec <- pmax(1.0, pmin(tau_vec, 1000.0))
    
    rho_base <- rnorm(N_GC, mean = top_cand$rho_base_mean, sd = top_cand$rho_base_sd)
    rho_base <- pmax(0.001, pmin(rho_base, 0.95))
    
    W_in <- rsparsematrix(N_GC, N_MF, density = top_cand$d_in)
    W_fb <- rsparsematrix(N_GoC, N_GC, density = top_cand$d_fb)
    fb_norm <- max(1e-6, suppressWarnings(tryCatch(norm(as.matrix(W_fb), "2"), error = function(e) 1.0)))
    W_fb <- W_fb * (top_cand$lambda_fb / fb_norm)
    
    W_inh <- rsparsematrix(N_GC, N_GoC, density = top_cand$d_inh)
    W_collateral <- rsparsematrix(N_GoC, 1 + N_actions, density = top_cand$d_collateral)
    
    model_obj <- new(ExactRModel, W_in, W_fb, W_inh, W_collateral, rho_base, tau_vec, N_actions, 0.05, 0.05, 1.5, 0.0)
    parent_models[[prot]] <- list(model = model_obj, params = top_cand)
  }
  
  test_datasets <- list()
  set.seed(999)
  for (prot in protocols) {
    test_datasets[[prot]] <- generate_pretraining_data(prot, T_steps = T_test, N_channels = N_MF, delta_t = 0.01)
  }
  
  G_matrix <- matrix(0, nrow = 4, ncol = 4, dimnames = list(Parent_Model = protocols, Target_Test = protocols))
  
  for (i_prot in protocols) {
    cat(sprintf("Evaluating Parent Model Trained on: %-10s ...\n", i_prot))
    model_i <- parent_models[[i_prot]]$model
    
    for (j_prot in protocols) {
      U_test_j <- test_datasets[[j_prot]]
      cv_score <- evaluate_shorthorizon_cv_prediction(model_i, U_test_j, k_max = 3, n_folds = 5, delta_t = 0.01)
      G_matrix[i_prot, j_prot] <- round(cv_score, 4)
    }
  }
  
  out_of_family_scores <- numeric(4)
  names(out_of_family_scores) <- protocols
  for (i in 1:4) {
    out_of_family_scores[i] <- mean(G_matrix[i, -i])
  }
  
  cat("\n==============================================================================\n")
  cat("SHORT-HORIZON TIME-SERIES CROSS-GENERALIZATION MATRIX (G_ij)\n")
  cat("==============================================================================\n")
  print(G_matrix)
  
  cat("\nOut-of-Family Generalization Index (Mean Transfer Accuracy):\n")
  print(round(out_of_family_scores, 4))
  
  best_gen_protocol <- names(which.max(out_of_family_scores))
  cat(sprintf("\nOptimal Pre-Training Protocol for Greatest Generalization: %s (Score: %.4f)\n",
              best_gen_protocol, out_of_family_scores[best_gen_protocol]))
  cat("==============================================================================\n")
  
  G_df <- as.data.frame(G_matrix)
  G_df$Generalization_Index <- round(out_of_family_scores, 4)
  write.csv(G_df, "cross_generalization_matrix.csv", row.names = TRUE)
  
  return(list(G_matrix = G_matrix, Generalization_Index = out_of_family_scores, Best_Protocol = best_gen_protocol))
}
