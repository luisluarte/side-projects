pacman::p_load(tidyverse, cmdstanr, loo, posterior)

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)), global_row = row_number())

start_idx <- dat_clean %>% group_by(participant_idx) %>% summarise(start = min(global_row), .groups="drop") %>% pull(start)
end_idx <- dat_clean %>% group_by(participant_idx) %>% summarise(end = max(global_row), .groups="drop") %>% pull(end)
min_rts <- dat_clean %>% group_by(participant_idx) %>% summarise(min_rt = min(RT), .groups="drop") %>% pull(min_rt)

stan_data <- list(
  N = nrow(dat_clean), S = length(unique(dat_clean$participant_idx)),
  start_idx = start_idx, end_idx = end_idx,
  choice = dat_clean$Boundary, rt = dat_clean$RT, reward = dat_clean$`F`, iti = dat_clean$ITI,
  min_rt = min_rts, grainsize = 1
)

cat('Compiling dynamic model...\n')
mod <- cmdstan_model('src/models/q_learning_ddm_dynamic.stan')

cat('Sampling...\n')
fit <- mod$sample(
  data = stan_data, 
  chains = 4, 
  parallel_chains = 4, 
  iter_warmup = 1000, 
  iter_sampling = 1000, 
  adapt_delta = 0.95,
  init = 0.1
)

fit$save_object('results/fit_dynamic_complete.rds')

cat('Compiling GQ model...\n')
mod_gq <- cmdstan_model('src/models/q_learning_ddm_dynamic_gq.stan')

cat('Running generated quantities...\n')
gq <- mod_gq$generate_quantities(fitted_params = fit, data = stan_data, parallel_chains = 4)

ll <- gq$draws('log_lik')
loo_dynamic <- loo(ll, r_eff = relative_eff(exp(ll)))

cat('\n--- LOO DYNAMIC MODEL ---\n')
print(loo_dynamic)

cat('\nLoading base model fit to compute its LOO...\n')
fit_base <- read_rds('results/fit_q_complete.rds')
mod_base_gq <- cmdstan_model('src/models/q_learning_ddm_gq.stan')
gq_base <- mod_base_gq$generate_quantities(fitted_params = fit_base, data = stan_data, parallel_chains = 4)
ll_base <- gq_base$draws('log_lik')
loo_base <- loo(ll_base, r_eff = relative_eff(exp(ll_base)))

cat('\n--- LOO BASE Q-LEARNING MODEL ---\n')
print(loo_base)

cat('\n--- COMPARISON (Dynamic - Base) ---\n')
print(loo_compare(loo_base, loo_dynamic))
