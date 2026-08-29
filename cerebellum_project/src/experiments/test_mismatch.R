
pacman::p_load(cmdstanr, posterior)
fit_bvk <- readRDS('../../results/fit_bvk_complete.rds')
mod_bvk_gq <- cmdstan_model('../models/bvk_continuous_gq.stan')
p_fit <- fit_bvk[['metadata']]()[['model_params']]

# We can't easily query mod_bvk_gq's expected parameters without dummy data.
# But we can inspect the CSV headers!
cat('Fitted params count:', length(p_fit), '\n')
cat(paste(p_fit[!grepl('\\[', p_fit)], collapse=', '), '\n')

