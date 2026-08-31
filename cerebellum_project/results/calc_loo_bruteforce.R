library(cmdstanr)
library(dplyr)
library(RWiener)
library(loo)

test_dat <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N10.rds")
N_subj <- max(test_dat$subj_idx)

fit_base <- readRDS("/home/DCCS5/cerebellum_project/results/baseline_urgency.rds")
fit_m009 <- readRDS("/home/DCCS5/cerebellum_project/results/m009_urgency.rds")

num_draws <- 100
draws_base <- fit_base$draws(format = "matrix")
draws_m009 <- fit_m009$draws(format = "matrix")
set.seed(42)
idx_draws <- sample(1:nrow(draws_base), num_draws)
draws_base <- draws_base[idx_draws, ]
draws_m009 <- draws_m009[idx_draws, ]

loglik_base <- matrix(0, nrow = num_draws, ncol = nrow(test_dat))
loglik_m009 <- matrix(0, nrow = num_draws, ncol = nrow(test_dat))

for(d in 1:num_draws) {
  for(s in 1:N_subj) {
    idx <- which(test_dat$subj_idx == s)
    subj_dat <- test_dat[idx, ]
    N_t <- nrow(subj_dat)
    
    a <- draws_base[d, paste0("a[",s,"]")]
    tnd <- draws_base[d, paste0("tnd[",s,"]")]
    v_ctx <- draws_base[d, paste0("v_ctx[",s,"]")]
    alpha_ctx <- draws_base[d, paste0("alpha_ctx[",s,"]")]
    
    Q <- c(0.5, 0.5)
    for(t in 1:N_t) {
      ch <- subj_dat$Boundary[t]
      R <- subj_dat$F[t]
      rt <- subj_dat$RT[t]
      
      veff <- if(ch==1) v_ctx * (Q[1]-Q[2]) else -v_ctx * (Q[1]-Q[2])
      
      resp_str <- if(ch==1) "upper" else "lower"
      if (rt <= tnd) rt_eff <- tnd + 0.001 else rt_eff <- rt
      
      den <- dwiener(rt_eff, alpha=a, tau=tnd, beta=0.5, delta=veff, resp=resp_str)
      if(is.na(den) || is.nan(den) || den < 1e-10) den <- 1e-10
      loglik_base[d, idx[t]] <- log(den)
      
      Q[ch] <- Q[ch] + alpha_ctx * (R - Q[ch])
    }
  }
}

frac_alpha <- 0.1 + 0.8 * (0:31)/31.0
kappa_vec <- 0.1 + 0.89 * (0:31)/31.0
inv_frac_alpha <- 1.0 - frac_alpha

for(d in 1:num_draws) {
  for(s in 1:N_subj) {
    idx <- which(test_dat$subj_idx == s)
    subj_dat <- test_dat[idx, ]
    N_t <- nrow(subj_dat)
    
    a_base_raw <- draws_m009[d, paste0("a_base_raw[",s,"]")]
    phys_a_base <- 0.11 + 3.0 * (1 / (1 + exp(-a_base_raw)))
    delta_max <- 1.0 / phys_a_base
    
    tnd <- draws_m009[d, paste0("tnd[",s,"]")]
    w_u <- draws_m009[d, paste0("w_u[",s,"]")]
    v_ctx <- draws_m009[d, paste0("v_ctx[",s,"]")] * 0.0540248
    gamma <- draws_m009[d, paste0("gamma_var[",s,"]")] * 0.0540248
    g_s <- draws_m009[d, paste0("golgi_scale[",s,"]")]
    
    a_c <- draws_m009[d, paste0("alpha_ctx[",s,"]")]
    a_pc <- draws_m009[d, paste0("alpha_pc[",s,"]")]
    inv_tau <- 1.0 / draws_m009[d, paste0("tau_decay[",s,"]")]
    
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
      rt <- subj_dat$RT[t]
      
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
      veff <- if(ch==1) veff_raw else -veff_raw
      
      U_epi <- sqrt((cb0^2 + 1e-8) * (cb1^2 + 1e-8))
      a_dyn <- phys_a_base + delta_max * tanh(w_u * U_epi)
      
      resp_str <- if(ch==1) "upper" else "lower"
      if (rt <= tnd) rt_eff <- tnd + 0.001 else rt_eff <- rt
      
      den <- dwiener(rt_eff, alpha=a_dyn, tau=tnd, beta=0.5, delta=veff, resp=resp_str)
      if(is.na(den) || is.nan(den) || den < 1e-10) den <- 1e-10
      loglik_m009[d, idx[t]] <- log(den)
      
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
}

cat("Evaluating LOO...\n")
loo_base <- suppressWarnings(loo(loglik_base))
loo_m009 <- suppressWarnings(loo(loglik_m009))

waic_base <- suppressWarnings(waic(loglik_base))
waic_m009 <- suppressWarnings(waic(loglik_m009))

cat("\n=== BASELINE METRICS ===\n")
cat("p_LOO: ", loo_base$estimates["p_loo", "Estimate"], "\n")
cat("p_WAIC: ", waic_base$estimates["p_waic", "Estimate"], "\n")
cat("ELPD_LOO: ", loo_base$estimates["elpd_loo", "Estimate"], "\n")

cat("\n=== M009 URGENCY METRICS ===\n")
cat("p_LOO: ", loo_m009$estimates["p_loo", "Estimate"], "\n")
cat("p_WAIC: ", waic_m009$estimates["p_waic", "Estimate"], "\n")
cat("ELPD_LOO: ", loo_m009$estimates["elpd_loo", "Estimate"], "\n")

cat("\n=== LOO COMPARISON (M009 vs Baseline) ===\n")
comp <- loo_compare(loo_base, loo_m009)
print(comp)