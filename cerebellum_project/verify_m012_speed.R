library(cmdstanr)
library(dplyr)

set.seed(42)
options(mc.cores = 8)

cat("==========================================\n")
cat(" VERIFYING M012 SPEED (100 ITERATIONS, N=30)\n")
cat("==========================================\n\n")

df_n30 <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N30.rds")
N_trials <- nrow(df_n30)
N_subj <- 30

start_idx <- integer(N_subj)
end_idx <- integer(N_subj)
for(s in 1:N_subj) {
  start_idx[s] <- min(which(df_n30$subj_idx == s))
  end_idx[s] <- max(which(df_n30$subj_idx == s))
}

stan_data_m012 <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = df_n30$subj_idx,
  resp = df_n30$Boundary,
  reward = df_n30$F,
  rt = df_n30$RT,
  iti = df_n30$ITI,
  min_rt = df_n30 %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m),
  start_idx = start_idx,
  end_idx = end_idx,
  grainsize = 4,
  W_exp = matrix(rnorm(N_subj * 4, 0, 1), nrow=N_subj, ncol=4),
  theta_mean = rep(0, 11),
  L_Sigma = diag(11)
)

mod_m012 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/m012_candidate.stan", cpp_options = list(stan_threads = TRUE))

t1 <- Sys.time()
fit_m012 <- mod_m012$sample(
  data = stan_data_m012,
  chains = 1,
  threads_per_chain = 8,
  init = 0,
  iter_warmup = 100,
  iter_sampling = 0,
  refresh = 1
)
t2 <- Sys.time()

elapsed <- as.numeric(difftime(t2, t1, units = "secs"))
cat("\nTotal Time for 50 Iterations (1 Chain, 8 Threads):", round(elapsed, 2), "seconds\n")
cat("Mean Time Per Iteration:", round(elapsed / 50, 3), "seconds\n")
cat("Projected Iterations Per Minute:", round(60 / (elapsed / 50), 2), "iters/min\n")
