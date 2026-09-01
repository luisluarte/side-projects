library(cmdstanr)
library(dplyr)
library(loo)

options(mc.cores = 8)

cat("Initializing Final Hardware-Optimized HMC Run for M012 Candidate...\n")
mod <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/m012_candidate.stan", cpp_options = list(stan_threads = TRUE))

df_n30 <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N30.rds")
df_n15 <- df_n30 %>% filter(subj_idx <= 15)
N_trials <- nrow(df_n15)
N_subj <- 15

set.seed(42)
W_exp <- matrix(rnorm(N_subj * 4, 0, 1), nrow=N_subj, ncol=4)
start_idx <- integer(N_subj)
end_idx <- integer(N_subj)
for(s in 1:N_subj) {
  start_idx[s] <- min(which(df_n15$subj_idx == s))
  end_idx[s] <- max(which(df_n15$subj_idx == s))
}

stan_data <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = df_n15$subj_idx,
  resp = df_n15$Boundary,
  reward = df_n15$F,
  rt = df_n15$RT,
  iti = df_n15$ITI,
  min_rt = df_n15 %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m),
  W_exp = W_exp,
  start_idx = start_idx,
  end_idx = end_idx,
  theta_mean = rep(0, 11),
  L_Sigma = diag(11),
  grainsize = 1
)

cat("\nStarting MCMC Sampling (init = 0, threads = 4, grainsize = 1)...\n")
fit <- mod$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 2,
  threads_per_chain = 4,
  init = 0,
  iter_warmup = 1000,
  iter_sampling = 1000,
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 10
)

cat("\nCalculating LOO Metrics...\n")
loo_m012 <- fit$loo()
saveRDS(loo_m012, "/home/DCCS5/cerebellum_project/results/loo_m012_candidate_n15.rds")
saveRDS(fit, "/home/DCCS5/cerebellum_project/results/fit_m012_candidate_n15.rds")

loo_v2 <- readRDS("/home/DCCS5/cerebellum_project/results/loo_v2_n15.rds")

cat("\n--- FINAL LOO COMPARISON (N=15) ---\n")
comp <- loo_compare(list(M012 = loo_m012, BaselineV2 = loo_v2))
print(comp)
