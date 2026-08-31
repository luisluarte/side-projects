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

options(mc.cores = 4)
mod_base_v2 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/baseline_v2.stan", 
                             cpp_options = list(stan_threads = TRUE))

cat("Starting HMC Fit for Baseline V2...\n")
fit_base_v2 <- mod_base_v2$sample(
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

fit_base_v2$save_object("/home/DCCS5/cerebellum_project/results/baseline_v2_urgency.rds")

cat("Extracting LOO for Baseline V2...\n")
log_lik_base_v2 <- fit_base_v2$draws("log_lik")
loo_base_v2 <- loo(log_lik_base_v2)

# Now, instead of re-running the manual R script for M009, 
# we can just use the exact native STAN log_lik for M009!
# We already uploaded m009_gq.stan but it had a mismatch because it didn't match perfectly.
# So for now, we just print the Baseline V2 LOO.
cat("\n=== BASELINE V2 METRICS ===\n")
print(loo_base_v2)

# Read the previously computed loo_m009 if it was saved? No, we computed it in memory.
# So we just print the baseline V2, we know M009 was ELPD: -1351.0