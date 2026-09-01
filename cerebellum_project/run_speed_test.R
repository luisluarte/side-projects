library(cmdstanr)
library(dplyr)

set.seed(42)
options(mc.cores = 8)

df_n30 <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N30.rds")
N_trials <- nrow(df_n30)
N_subj <- 30

stan_data <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = df_n30$subj_idx,
  resp = df_n30$Boundary,
  reward = df_n30$F,
  rt = df_n30$RT,
  iti = df_n30$ITI,
  min_rt = df_n30 %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m),
  grainsize = 1
)

cat("Testing Baseline V1 Speed (10 Iterations)...\n")
mod_v1 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/baseline.stan")
t1 <- Sys.time()
fit_v1 <- mod_v1$sample(
  data = stan_data,
  chains = 1,
  init = 0,
  iter_warmup = 10,
  iter_sampling = 0,
  refresh = 1
)
t2 <- Sys.time()
cat("Baseline V1 10-Iteration Time: ", round(as.numeric(difftime(t2, t1, units = "secs")), 2), " seconds\n\n")

cat("Testing Baseline V2 Speed (10 Iterations)...\n")
mod_v2 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/baseline_v2.stan", cpp_options = list(stan_threads = TRUE))
t3 <- Sys.time()
fit_v2 <- mod_v2$sample(
  data = stan_data,
  chains = 1,
  threads_per_chain = 8,
  init = 0,
  iter_warmup = 10,
  iter_sampling = 0,
  refresh = 1
)
t4 <- Sys.time()
cat("Baseline V2 10-Iteration Time: ", round(as.numeric(difftime(t4, t3, units = "secs")), 2), " seconds\n\n")
