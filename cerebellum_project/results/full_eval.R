library(cmdstanr)
library(dplyr)
library(tidyr)
library(pROC)
library(Metrics)

test_dat <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N10.rds")
N_subj <- max(test_dat$subj_idx)

fit_base <- readRDS("/home/DCCS5/cerebellum_project/results/baseline_urgency.rds")
fit_m009 <- readRDS("/home/DCCS5/cerebellum_project/results/m009_urgency.rds")

base_preds <- numeric(nrow(test_dat))
base_erts <- numeric(nrow(test_dat))
m009_preds <- numeric(nrow(test_dat))
m009_erts <- numeric(nrow(test_dat))
labels <- numeric(nrow(test_dat))

base_tnd <- fit_base$summary("tnd", "mean")$mean
base_v <- fit_base$summary("v_ctx", "mean")$mean
base_a_phys <- fit_base$summary("a", "mean")$mean

for(s in 1:N_subj) {
  idx <- which(test_dat$subj_idx == s)
  labels[idx] <- ifelse(test_dat$Boundary[idx] == 1, 1, 0)
  
  v <- base_v[s]
  a <- base_a_phys[s]
  tnd <- base_tnd[s]
  
  p_upper <- 1.0 / (1.0 + exp(-v * a))
  base_preds[idx] <- p_upper
  
  abs_v <- abs(v)
  if(abs_v < 1e-4) abs_v <- 1e-4
  base_erts[idx] <- tnd + (a / (2*abs_v)) * tanh(a * abs_v / 2)
}

m009_tnd <- fit_m009$summary("tnd", "mean")$mean
m009_a_raw <- fit_m009$summary("a_base_raw", "mean")$mean
m009_w_u <- fit_m009$summary("w_u", "mean")$mean
m009_v_ctx <- fit_m009$summary("v_ctx", "mean")$mean
m009_gamma <- fit_m009$summary("gamma_var", "mean")$mean
m009_g_s <- fit_m009$summary("golgi_scale", "mean")$mean
m009_alpha_ctx <- fit_m009$summary("alpha_ctx", "mean")$mean
m009_alpha_pc <- fit_m009$summary("alpha_pc", "mean")$mean
m009_tau_decay <- fit_m009$summary("tau_decay", "mean")$mean

frac_alpha <- 0.1 + 0.8 * (0:31)/31.0
kappa_vec <- 0.1 + 0.89 * (0:31)/31.0
inv_frac_alpha <- 1.0 - frac_alpha

for(s in 1:N_subj) {
  idx <- which(test_dat$subj_idx == s)
  subj_dat <- test_dat[idx, ]
  N_t <- nrow(subj_dat)
  
  a_base_raw <- m009_a_raw[s]
  phys_a_base <- 0.11 + 3.0 * (1 / (1 + exp(-a_base_raw)))
  delta_max <- 1.0 / phys_a_base
  
  tnd <- m009_tnd[s]
  w_u <- m009_w_u[s]
  v_ctx <- m009_v_ctx[s] * 0.0540248
  gamma <- m009_gamma[s] * 0.0540248
  g_s <- m009_g_s[s]
  
  a_c <- m009_alpha_ctx[s]
  a_pc <- m009_alpha_pc[s]
  inv_tau <- 1.0 / m009_tau_decay[s]
  
  set.seed(42)
  W_exp <- rnorm(32, 0, 1)
  inv_W_exp <- inv_frac_alpha * W_exp
  
  Q <- c(0.5, 0.5)
  Q_diff <- 0.0
  frac_mem <- rep(0.0, 32)
  Z <- rep(0.0, 32)
  W_PC <- rep(0.0, 32)
  
  for(t in 1:N_t) {
    ch <- subj_dat$Boundary[t]
    R <- subj_dat$F[t]
    iti <- subj_dat$ITI[t]
    
    phys_decay <- exp(-iti * inv_tau)
    frac_mem <- frac_alpha * frac_mem + inv_W_exp * Q[ch]
    Z <- phys_decay * (kappa_vec * Z) + tanh(frac_mem)
    
    W_PC_eff <- 3.0 * tanh(W_PC / 3.0)
    eff_z <- W_PC_eff * Z
    abs_approx <- sqrt(eff_z^2 + 1e-8)
    S_mask <- tanh(g_s * abs_approx)
    
    cb0 <- sum(S_mask[1:16] * eff_z[1:16])
    cb1 <- sum(S_mask[17:32] * eff_z[17:32])
    
    veff_scaled <- v_ctx * Q_diff + gamma * (cb0 - cb1)
    veff_raw <- 18.51 * tanh(veff_scaled)
    
    U_epi <- sqrt((cb0^2 + 1e-8) * (cb1^2 + 1e-8))
    a_dyn <- phys_a_base + delta_max * tanh(w_u * U_epi)
    
    p_upper <- 1.0 / (1.0 + exp(-veff_raw * a_dyn))
    m009_preds[idx[t]] <- p_upper
    
    abs_v <- abs(veff_raw)
    if(abs_v < 1e-4) abs_v <- 1e-4
    m009_erts[idx[t]] <- tnd + (a_dyn / (2 * abs_v)) * tanh(a_dyn * abs_v / 2)
    
    prev_E <- R - Q[ch]
    a_ctx_E <- a_c * prev_E
    Q[ch] <- Q[ch] + a_ctx_E
    Q_diff <- Q_diff + (ifelse(ch==1, -1, 1) * a_ctx_E)
    
    if(ch == 1) {
      W_PC[1:16] <- W_PC[1:16] + a_pc * prev_E * Z[1:16]
    } else {
      W_PC[17:32] <- W_PC[17:32] + a_pc * prev_E * Z[17:32]
    }
  }
}

calc_mcc <- function(y_true, y_prob) {
  y_pred <- ifelse(y_prob > 0.5, 1, 0)
  TP <- sum(y_true == 1 & y_pred == 1)
  TN <- sum(y_true == 0 & y_pred == 0)
  FP <- sum(y_true == 0 & y_pred == 1)
  FN <- sum(y_true == 1 & y_pred == 0)
  num <- (TP * TN) - (FP * FN)
  den <- sqrt(as.numeric(TP + FP) * as.numeric(TP + FN) * as.numeric(TN + FP) * as.numeric(TN + FN))
  if (den == 0) return(0)
  return(num / den)
}

calc_prauc <- function(y_true, y_prob) {
  ord <- order(y_prob, decreasing = TRUE)
  y_true <- y_true[ord]
  tp <- cumsum(y_true)
  fp <- cumsum(!y_true)
  prec <- tp / (tp + fp)
  rec <- tp / sum(y_true)
  
  # Area under curve
  dx <- diff(rec)
  my_prec <- (prec[-1] + prec[-length(prec)]) / 2
  sum(dx * my_prec)
}

cat("\n=== BASELINE METRICS ===\n")
cat("ROC-AUC: ", as.numeric(roc(labels, base_preds, quiet=TRUE)$auc), "\n")
cat("PR-AUC:  ", calc_prauc(labels, base_preds), "\n")
cat("MCC:     ", calc_mcc(labels, base_preds), "\n")
cat("RT-RMSE: ", rmse(test_dat$RT, base_erts), "\n")

cat("\n=== M009 URGENCY METRICS ===\n")
cat("ROC-AUC: ", as.numeric(roc(labels, m009_preds, quiet=TRUE)$auc), "\n")
cat("PR-AUC:  ", calc_prauc(labels, m009_preds), "\n")
cat("MCC:     ", calc_mcc(labels, m009_preds), "\n")
cat("RT-RMSE: ", rmse(test_dat$RT, m009_erts), "\n")