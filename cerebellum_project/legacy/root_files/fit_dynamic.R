pacman::p_load(tidyverse, cmdstanr, loo, posterior)

cat('Loading stan_data...\n')
stan_data <- read_rds('results/stan_data.rds')

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
