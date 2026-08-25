pacman::p_load(tidyverse, cmdstanr, loo, posterior)

cat("loading fits...\n")
fit_base <- read_rds("results/fit_q_complete.rds")
fit_gating <- read_rds("results/fit_full_gating_complete.rds")

stan_data <- read_rds("results/stan_data.rds")

mod_base_gq <- cmdstan_model("src/models/q_learning_ddm_gq.stan")
gq_base <- mod_base_gq$generate_quantities(fitted_params = fit_base, data = stan_data, parallel_chains = 4)
ll_base <- gq_base$draws("log_lik")
loo_base <- loo(ll_base, r_eff = relative_eff(exp(ll_base)))

mod_gating_gq <- cmdstan_model("src/models/bvk_full_gating_gq.stan")
gq_gating <- mod_gating_gq$generate_quantities(fitted_params = fit_gating, data = stan_data, parallel_chains = 4)
ll_gating <- gq_gating$draws("log_lik")
loo_gating <- loo(ll_gating, r_eff = relative_eff(exp(ll_gating)))

print(loo_base)
print(loo_gating)

ll_mat_base <- as_draws_matrix(ll_base)
ll_mat_gating <- as_draws_matrix(ll_gating)
library(matrixStats)
cat("Sum colLogSumExps Base: ", sum(colLogSumExps(ll_mat_base) - log(nrow(ll_mat_base))), "\n")
cat("Sum colLogSumExps Gating: ", sum(colLogSumExps(ll_mat_gating) - log(nrow(ll_mat_gating))), "\n")
