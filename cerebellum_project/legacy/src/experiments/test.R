
pacman::p_load(cmdstanr, posterior, tidyverse)
fit_bvk <- readRDS('../../results/fit_bvk_complete.rds')

# Just look at the metadata! CmdStanMCMC objects have data embedded if saved properly, or at least they might have min_rt as data?
# Let's extract the time or sizes
print(fit_bvk[['metadata']]()[['stan_variable_sizes']])

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

# Load raw data to match
dat_raw <- read_csv('../../data/raw/behavioral_compilate.csv', show_col_types = FALSE)
dat_clean <- dat_raw %>%
  arrange(participant_id, ttp) %>%
  group_by(participant_id) %>%
  mutate(RT = (ttr - ttp) / 1000) %>%
  ungroup() %>%
  filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(F))

subject_counts <- dat_clean %>% group_by(participant_id) %>% summarise(min_rt = min(RT)) %>% ungroup()

# Match the min_rts
found_pids <- character(30)
for(s in 1:30) {
  # Find closest min_rt in the empirical data
  target <- min_rt_reconstructed[s]
  dists <- abs(subject_counts[['min_rt']] - target)
  best_idx <- which.min(dists)
  found_pids[s] <- subject_counts[['participant_id']][best_idx]
}

# Print the found pids!
dput(found_pids)

