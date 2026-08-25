
pacman::p_load(tidyverse, cmdstanr, posterior, loo)
fit_bvk <- readRDS('../../results/fit_bvk_complete.rds')

draws <- as_draws_df(fit_bvk[['draws']](variables = c('tau_nd', 'mu_tau_nd', 'sigma_tau_nd', 'z_tau_nd')))
min_rt_reconstructed <- numeric(30)
for (s in 1:30) {
  tau_nd_s <- draws[[paste0('tau_nd[', s, ']')]]
  mu_tau_nd <- draws[['mu_tau_nd']]
  sigma_tau_nd <- draws[['sigma_tau_nd']]
  z_tau_nd_s <- draws[[paste0('z_tau_nd[', s, ']')]]
  inv_logit_val <- plogis(mu_tau_nd + sigma_tau_nd * z_tau_nd_s)
  min_rt_s <- (tau_nd_s - 0.001) / inv_logit_val + 0.002
  min_rt_reconstructed[s] <- mean(min_rt_s)
}

dat_raw <- read_csv('../../data/raw/behavioral_compilate.csv', show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(F))
subject_counts <- dat_clean %>% group_by(participant_id) %>% summarise(min_rt = min(RT)) %>% ungroup()
found_pids <- character(30)
for(s in 1:30) {
  target <- min_rt_reconstructed[s]
  dists <- abs(subject_counts[['min_rt']] - target)
  best_idx <- which.min(dists)
  found_pids[s] <- subject_counts[['participant_id']][best_idx]
}

dat_reconstructed <- dat_clean %>% filter(participant_id %in% found_pids) %>% group_by(participant_id) %>% mutate(ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup()
rescued_counts <- dat_reconstructed %>% group_by(participant_id) %>% summarise(count = n(), min_rt = min(RT)) %>% ungroup()
end_idx <- cumsum(rescued_counts[['count']])
start_idx <- end_idx - rescued_counts[['count']] + 1

stan_data <- list(N = nrow(dat_reconstructed), S = nrow(rescued_counts), start_idx = start_idx, end_idx = end_idx, choice = dat_reconstructed[['Boundary']], rt = dat_reconstructed[['RT']], reward = dat_reconstructed[['F']], iti = dat_reconstructed[['ITI']], f_dur = dat_reconstructed[['F_dur']], min_rt = rescued_counts[['min_rt']], N_MF = 5, grainsize = 1)

mod_bvk_gq <- cmdstan_model('../models/bvk_continuous_gq.stan')
gq_bvk <- mod_bvk_gq[['generate_quantities']](fitted_params = fit_bvk, data = stan_data, parallel_chains = 4)
ll_bvk <- gq_bvk[['draws']]('log_lik', format = 'array')

# Check raw variances
vars <- apply(ll_bvk, 3, var)
cat('Min var:', min(vars), '\n')
cat('Count < 1e-12:', sum(vars < 1e-12), '\n')
cat('Count < 1e-6:', sum(vars < 1e-6), '\n')
cat('Count < 1e-4:', sum(vars < 1e-4), '\n')

# If I add noise to EVERYTHING
ll_bvk_noise <- ll_bvk + array(rnorm(length(ll_bvk), 0, 1e-5), dim(ll_bvk))

r_eff_bvk <- relative_eff(exp(ll_bvk_noise))
r_eff_bvk[is.na(r_eff_bvk)] <- 1.0
loo_bvk <- loo(ll_bvk_noise, r_eff = r_eff_bvk)
print(loo_bvk)

