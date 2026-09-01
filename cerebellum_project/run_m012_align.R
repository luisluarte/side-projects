library(cmdstanr)
library(dplyr)
library(readr)
library(loo)

options(mc.cores = 8)

cat("Compiling M012_ALIGN...\n")
mod <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/m012_align.stan", cpp_options = list(stan_threads = TRUE))

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

theta_mean <- rep(0, 11)
L_Sigma <- diag(11)

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
  theta_mean = theta_mean,
  L_Sigma = L_Sigma
)

cat("\nRunning Pathfinder for M012_ALIGN...\n")
pf <- mod$pathfinder(
  data = stan_data,
  num_paths = 4,
  single_path_draws = 40,
  history_size = 15,
  max_lbfgs_iters = 100
)

cat("\nStarting HMC (initialized via Pathfinder)...\n")
fit <- mod$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  init = pf,
  iter_warmup = 1000,
  iter_sampling = 1000,
  max_treedepth = 10,
  adapt_delta = 0.8,
  refresh = 100
)

loo_m012 <- fit$loo()
saveRDS(loo_m012, "/home/DCCS5/cerebellum_project/results/loo_m012_align_n15.rds")

loo_v2 <- readRDS("/home/DCCS5/cerebellum_project/results/loo_v2_n15.rds")

cat("\n--- LOO COMPARISON (N=15) ---\n")
comp <- loo_compare(list(M012_Align = loo_m012, BaselineV2 = loo_v2))
print(comp)
