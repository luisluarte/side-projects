library(cmdstanr)
library(dplyr)
library(posterior)

cat("Loading Models and Data...\n")
df_n30 <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N30.rds")
# Use N=5
df_n5 <- df_n30 %>% filter(subj_idx <= 5)
df_n5 <- df_n5 %>% group_by(subj_idx) %>% mutate(seq_t = row_number()) %>% ungroup()
# Re-index subjects
df_n5$subj_new <- match(df_n5$subj_idx, unique(df_n5$subj_idx))
N_subj <- length(unique(df_n5$subj_new))
N_trials <- nrow(df_n5)

start_idx <- integer(N_subj)
end_idx <- integer(N_subj)
for(s in 1:N_subj) {
  idx <- which(df_n5$subj_new == s)
  start_idx[s] <- min(idx)
  end_idx[s] <- max(idx)
}
min_rt <- df_n5 %>% group_by(subj_new) %>% summarize(min_rt = min(RT)) %>% pull(min_rt)

W_exp <- matrix(0, nrow=N_subj, ncol=4)
theta_mean_vopt <- rep(0, 8)
L_Sigma_vopt <- diag(8)
theta_mean_m012 <- rep(0, 12)
L_Sigma_m012 <- diag(12)

stan_data_vopt <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = df_n5$subj_new,
  resp = df_n5$Boundary,
  reward = df_n5$F,
  rt = df_n5$RT,
  min_rt = min_rt,
  start_idx = start_idx,
  end_idx = end_idx,
  theta_mean = theta_mean_vopt,
  L_Sigma = L_Sigma_vopt,
  grainsize = 1
)

stan_data_m012 <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = df_n5$subj_new,
  resp = df_n5$Boundary,
  reward = df_n5$F,
  rt = df_n5$RT,
  iti = df_n5$ITI,
  min_rt = min_rt,
  start_idx = start_idx,
  end_idx = end_idx,
  W_exp = W_exp,
  theta_mean = theta_mean_m012,
  L_Sigma = L_Sigma_m012,
  grainsize = 1
)

cat("Compiling Models...\n")
mod_vopt <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/vopt_ss3.stan", cpp_options = list(stan_threads = TRUE))
mod_m012 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/m012_ss3.stan", cpp_options = list(stan_threads = TRUE))

cat("Fitting Models...\n")
fit_vopt <- mod_vopt$sample(data = stan_data_vopt, chains = 4, parallel_chains = 4, threads_per_chain = 2, iter_warmup = 300, iter_sampling = 300, refresh = 100, init = 0)
fit_m012 <- mod_m012$sample(data = stan_data_m012, chains = 4, parallel_chains = 4, threads_per_chain = 2, iter_warmup = 300, iter_sampling = 300, refresh = 100, init = 0)

cat("Computing Confusion Matrices...\n")
draws_vopt <- as_draws_df(fit_vopt$draws())
draws_m012 <- as_draws_df(fit_m012$draws())

get_median <- function(fit, prefix, n_subj) {
  unlist(sapply(1:n_subj, function(i) median(fit[[paste0(prefix, "[", i, "]")]])))
}

v_abase <- get_median(draws_vopt, "a_base_raw", 5)
v_vctx <- get_median(draws_vopt, "v_ctx", 5)
v_wbias <- get_median(draws_vopt, "w_bias_raw", 5)
v_aw <- get_median(draws_vopt, "aw", 5)
v_al <- get_median(draws_vopt, "al", 5)
v_wctx <- get_median(draws_vopt, "w_ctx", 5)
v_betamis <- get_median(draws_vopt, "beta_mismatch", 5)

m_abase <- get_median(draws_m012, "a_base_raw", 5)
m_vctx <- get_median(draws_m012, "v_ctx", 5)
m_wbias <- get_median(draws_m012, "w_bias_raw", 5)
m_aw <- get_median(draws_m012, "aw", 5)
m_al <- get_median(draws_m012, "al", 5)
m_tau <- get_median(draws_m012, "tau_decay", 5)
m_alphaPC <- get_median(draws_m012, "alpha_pc", 5)
m_wctx <- get_median(draws_m012, "w_ctx", 5)
m_wcb <- get_median(draws_m012, "w_cb", 5)
m_betamis <- get_median(draws_m012, "beta_mismatch", 5)
m_g <- get_median(draws_m012, "golgi_scale", 5)

m_frac <- 0.1 + 0.8 * (0:3 / 3.0)
m_inv_frac <- 1.0 - m_frac
m_kappa <- 0.1 + 0.89 * (0:3 / 3.0)

res_list <- list()

for(s in 1:5) {
  d_s <- df_n5 %>% filter(subj_new == s)
  n <- nrow(d_s)
  
  Q_v <- c(0.5, 0.5)
  Q_m <- c(0.5, 0.5)
  W_exp_s <- rep(0.0, 4)
  frac_mem <- rep(0.0, 4)
  Z <- rep(0.0, 4)
  W_PC_latent <- rep(0.0, 4)
  
  phys_a_vopt <- 0.11 + 3.0 * exp(v_abase[s]) / (1 + exp(v_abase[s]))
  delta_max_vopt <- 1.0 / phys_a_vopt
  phys_a_m012 <- 0.11 + 3.0 * exp(m_abase[s]) / (1 + exp(m_abase[s]))
  delta_max_m012 <- 1.0 / phys_a_m012
  
  pred_sw_vopt <- numeric(n)
  pred_sw_m012 <- numeric(n)
  actual_sw <- numeric(n)
  
  for(t in 1:n) {
    ch <- as.integer(d_s$Boundary[t])
    R <- as.numeric(d_s$F[t])
    
    if(t > 1) {
      prev_ch <- as.integer(d_s$Boundary[t-1])
      is_switch <- ifelse(ch != prev_ch, 1, 0)
      actual_sw[t] <- is_switch
      
      # V-OPT
      Q_diff_v <- Q_v[1] - Q_v[2]
      M_align_v <- tanh(v_wctx[s] * Q_diff_v)
      caution_v <- log1p(exp(-5.0 * abs(M_align_v))) * 0.1
      a_dyn_v <- phys_a_vopt + delta_max_vopt * tanh(v_betamis[s] * caution_v)
      
      Q_switch_v <- Q_v[3 - prev_ch]
      Q_stay_v <- Q_v[prev_ch]
      w_start_v <- plogis(v_wbias[s])
      veff_v <- v_vctx[s] * (Q_switch_v - Q_stay_v)
      if (veff_v == 0) {
        pred_sw_vopt[t] <- w_start_v
      } else {
        pred_sw_vopt[t] <- (exp(-2 * veff_v * a_dyn_v * w_start_v) - 1) / (exp(-2 * veff_v * a_dyn_v) - 1)
      }
      
      # M012
      Q_diff_m <- Q_m[1] - Q_m[2]
      W_PC_eff <- 3.0 * tanh(W_PC_latent * 0.33333333)
      eff_z <- W_PC_eff * Z
      abs_approx <- sqrt(eff_z * eff_z + 1e-8)
      S_mask <- tanh(m_g[s] * abs_approx)
      
      cb0 <- sum(S_mask[1:2] * eff_z[1:2])
      cb1 <- sum(S_mask[3:4] * eff_z[3:4])
      Cb_diff <- cb0 - cb1
      
      M_align_m <- tanh(m_wcb[s] * Cb_diff) * tanh(m_wctx[s] * Q_diff_m)
      caution_m <- log1p(exp(-10.0 * M_align_m)) * 0.1
      a_dyn_m <- phys_a_m012 + delta_max_m012 * tanh(m_betamis[s] * caution_m)
      
      Q_switch_m <- Q_m[3 - prev_ch]
      Q_stay_m <- Q_m[prev_ch]
      w_start_m <- plogis(m_wbias[s])
      veff_m <- m_vctx[s] * (Q_switch_m - Q_stay_m)
      if (veff_m == 0) {
        pred_sw_m012[t] <- w_start_m
      } else {
        pred_sw_m012[t] <- (exp(-2 * veff_m * a_dyn_m * w_start_m) - 1) / (exp(-2 * veff_m * a_dyn_m) - 1)
      }
    }
    
    # Update VOPT
    pe_v <- R - Q_v[ch]
    Q_v[ch] <- Q_v[ch] + ifelse(pe_v > 0, v_aw[s], v_al[s]) * pe_v
    
    # Update M012
    iti_s <- ifelse(d_s$ITI[t] < 0, 1.0, d_s$ITI[t])
    phys_decay <- exp(-iti_s / m_tau[s])
    Q_m <- 0.5 + (Q_m - 0.5) * phys_decay
    
    inv_W_exp <- m_inv_frac * W_exp_s
    frac_mem <- m_frac * frac_mem + inv_W_exp * Q_m[ch]
    Z <- phys_decay * (m_kappa * Z) + tanh(frac_mem)
    
    pe_m <- R - Q_m[ch]
    Q_m[ch] <- Q_m[ch] + ifelse(pe_m > 0, m_aw[s], m_al[s]) * pe_m
    
    alpha_E <- m_alphaPC[s] * pe_m
    if(ch == 1) {
      W_PC_latent[1:2] <- W_PC_latent[1:2] + alpha_E * Z[1:2]
    } else {
      W_PC_latent[3:4] <- W_PC_latent[3:4] + alpha_E * Z[3:4]
    }
  }
  
  d_s$pred_sw_vopt <- pred_sw_vopt
  d_s$pred_sw_m012 <- pred_sw_m012
  d_s$actual_sw <- actual_sw
  res_list[[s]] <- d_s
}

df_res <- bind_rows(res_list) %>% filter(seq_t > 1)

df_res$vopt_call <- factor(ifelse(df_res$pred_sw_vopt > 0.24, "Pred Switch", "Pred Stay"), levels=c("Pred Stay", "Pred Switch"))
df_res$m012_call <- factor(ifelse(df_res$pred_sw_m012 > 0.24, "Pred Switch", "Pred Stay"), levels=c("Pred Stay", "Pred Switch"))
df_res$actual_label <- factor(ifelse(df_res$actual_sw == 1, "Actual Switch", "Actual Stay"), levels=c("Actual Stay", "Actual Switch"))

sink("/home/DCCS5/cerebellum_project/results/conf_ss3.txt")
cat("\n=== V-OPT SS3 Confusion Matrix ===\n")
t_vopt <- table(Actual = df_res$actual_label, Predicted = df_res$vopt_call)
print(t_vopt)
cat("\nRow-Normalized (Recall):\n")
print(round(prop.table(t_vopt, 1) * 100, 1))

cat("\n=== M012 SS3 Confusion Matrix ===\n")
t_m012 <- table(Actual = df_res$actual_label, Predicted = df_res$m012_call)
print(t_m012)
cat("\nRow-Normalized (Recall):\n")
print(round(prop.table(t_m012, 1) * 100, 1))

loo_vopt <- fit_vopt$loo()
loo_m012 <- fit_m012$loo()
cat("\n=== ELPD Comparison ===\n")
print(loo::loo_compare(list(vopt = loo_vopt, m012 = loo_m012)))
sink()
