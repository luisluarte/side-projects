
pacman::p_load(tidyverse, cmdstanr, posterior, loo)
fit_bvk <- readRDS('../../results/fit_bvk_complete.rds')

# We can extract the log_lik if it was saved in the RDS!
# Wait! Does fit_bvk_complete.rds contain 'log_lik' already?
p_names <- fit_bvk[['metadata']]()[['model_params']]
cat('log_lik in params?', any(grepl('log_lik', p_names)), '\n')

# What about variables?
draws_names <- fit_bvk[['metadata']]()[['stan_variables']]
cat('log_lik in variables?', any(grepl('log_lik', draws_names)), '\n')

