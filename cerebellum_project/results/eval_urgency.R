library(cmdstanr)
library(dplyr)
library(tidyr)
library(readr)
library(Metrics)

test_dat <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N10.rds")

# Load models
fit_base <- readRDS("/home/DCCS5/cerebellum_project/results/baseline_urgency.rds")
fit_m009 <- readRDS("/home/DCCS5/cerebellum_project/results/m009_urgency.rds")

N_subj <- max(test_dat$subj_idx)

# Baseline Evaluation
base_tnd <- fit_base$summary("tnd", "mean")$mean
base_v <- fit_base$summary("v_ctx", "mean")$mean
base_a_phys <- fit_base$summary("a", "mean")$mean

base_rmse <- numeric(N_subj)
for(s in 1:N_subj) {
  subj_dat <- test_dat %>% filter(subj_idx == s)
  v <- base_v[s]
  a <- base_a_phys[s]
  tnd <- base_tnd[s]
  
  abs_v <- abs(v)
  if(abs_v < 1e-4) abs_v <- 1e-4
  expected_rt <- tnd + (a / (2*abs_v)) * tanh(a * abs_v / 2)
  
  base_rmse[s] <- rmse(subj_dat$RT, expected_rt)
}

# M009 Urgency Evaluation
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

m009_rmse <- numeric(N_subj)
for(s in 1:N_subj) {
  subj_dat <- test_dat %>% filter(subj_idx == s)
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
  
  expected_rts <- numeric(N_t)
  
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
    abs_v <- abs(18.51 * tanh(veff_scaled))
    if(abs_v < 1e-4) abs_v <- 1e-4
    
    U_epi <- sqrt((cb0^2 + 1e-8) * (cb1^2 + 1e-8))
    a_dyn <- phys_a_base + delta_max * tanh(w_u * U_epi)
    
    expected_rts[t] <- tnd + (a_dyn / (2 * abs_v)) * tanh(a_dyn * abs_v / 2)
    
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
  
  m009_rmse[s] <- rmse(subj_dat$RT, expected_rts)
}

cat("=== RT-RMSE RESULTS (N=10) ===\n")
cat("Baseline Mean RMSE: ", mean(base_rmse), "\n")
cat("M009 Urgency Mean RMSE: ", mean(m009_rmse), "\n")