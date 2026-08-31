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
  min_rt = test_dat %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m)
)

options(mc.cores = 4)
mod_base_v3 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/baseline_v3.stan", 
                             cpp_options = list(stan_threads = TRUE))

cat("Starting HMC Fit for Baseline V3...\n")
fit_base_v3 <- mod_base_v3$sample(
  data = stan_data,
  seed = 42,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 7,
  iter_warmup = 400,
  iter_sampling = 300,
  refresh = 100,
  max_treedepth = 10
)

cat("Extracting LOO for Baseline V3...\n")
log_lik_base_v3 <- fit_base_v3$draws("log_lik")
loo_base_v3 <- loo(log_lik_base_v3)

cat("\n=== BASELINE V3 METRICS ===\n")
print(loo_base_v3)