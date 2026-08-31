library(cmdstanr)
library(dplyr)
library(readr)
library(loo)

test_dat <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N10.rds")

stan_data <- list(
  N_trials = nrow(test_dat),
  N_subj = max(test_dat$subj_idx),
  subj = test_dat$subj_idx,
  resp = test_dat$Boundary,
  reward = test_dat$F,
  rt = test_dat$RT,
  iti = test_dat$ITI
)

min_rt <- test_dat %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m)
stan_data$min_rt <- min_rt

start_idx <- numeric(stan_data$N_subj)
end_idx <- numeric(stan_data$N_subj)
for(s in 1:stan_data$N_subj) {
  idx <- which(test_dat$subj_idx == s)
  start_idx[s] <- min(idx)
  end_idx[s] <- max(idx)
}
stan_data$start_idx <- start_idx
stan_data$end_idx <- end_idx
set.seed(42)
stan_data$W_exp <- matrix(rnorm(stan_data$N_subj * 32, 0, 1), nrow=stan_data$N_subj, ncol=32)
stan_data$theta_mean <- c(1.36, -0.63, 0.44, -1.94, -0.06, -0.09, 1.48, 1.05, -0.58)
cov_matrix <- matrix(c(
  0.13, 0.04, 0.01, 0.03, 0.00, 0.04, -0.01, -0.02, 0.01,
  0.04, 0.17, 0.02, 0.05, 0.01, 0.02, 0.01, -0.03, -0.02,
  0.01, 0.02, 0.22, 0.01, 0.02, 0.03, 0.05, 0.02, 0.01,
  0.03, 0.05, 0.01, 0.11, -0.02, -0.01, 0.03, -0.01, 0.00,
  0.00, 0.01, 0.02, -0.02, 0.08, 0.04, 0.02, -0.01, 0.01,
  0.04, 0.02, 0.03, -0.01, 0.04, 0.15, 0.04, 0.03, -0.02,
  -0.01, 0.01, 0.05, 0.03, 0.02, 0.04, 0.14, -0.01, 0.02,
  -0.02, -0.03, 0.02, -0.01, -0.01, 0.03, -0.01, 0.12, 0.00,
  0.01, -0.02, 0.01, 0.00, 0.01, -0.02, 0.02, 0.00, 0.09
), nrow=9, byrow=TRUE)
stan_data$L_Sigma <- t(chol(cov_matrix))

fit_base <- readRDS("/home/DCCS5/cerebellum_project/results/baseline_urgency.rds")
fit_m009 <- readRDS("/home/DCCS5/cerebellum_project/results/m009_urgency.rds")

mod_base_gq <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/baseline_gq.stan")
gq_base <- mod_base_gq$generate_quantities(fit_base, data = stan_data, parallel_chains=4)
log_lik_base <- gq_base$draws("log_lik")
loo_base <- loo(log_lik_base)

mod_m009_gq <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/m009_gq.stan")
gq_m009 <- mod_m009_gq$generate_quantities(fit_m009, data = stan_data, parallel_chains=4)
log_lik_m009 <- gq_m009$draws("log_lik")
loo_m009 <- loo(log_lik_m009)

cat("\n=== BASELINE LOO ===\n")
print(loo_base)

cat("\n=== M009 URGENCY LOO ===\n")
print(loo_m009)

cat("\n=== MODEL COMPARISON ===\n")
print(loo_compare(loo_base, loo_m009))