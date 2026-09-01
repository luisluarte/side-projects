library(cmdstanr)
library(dplyr)
library(loo)

options(mc.cores = 8)

cat("Compiling m012_pathfinder_test.stan...\n")
mod <- cmdstan_model("C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/src/stan/m012_pathfinder_test.stan", cpp_options = list(stan_threads = TRUE))

N_subj <- 15
trials_per_subj <- 100
N_trials <- N_subj * trials_per_subj

df_n15 <- data.frame(
  subj_idx = rep(1:N_subj, each = trials_per_subj),
  Boundary = sample(1:2, N_trials, replace = TRUE),
  F = runif(N_trials, 0, 1),
  RT = runif(N_trials, 0.2, 1.5),
  ITI = runif(N_trials, 0.5, 2.0)
)

set.seed(42)
W_exp <- matrix(rnorm(N_subj * 32, 0, 1), nrow=N_subj, ncol=32)

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

cat("\nRunning Pathfinder for m012_pathfinder_test.stan...\n")
pf <- mod(
  data = stan_data,
  num_paths = 4,
  single_path_draws = 40,
  history_size = 15,
  max_lbfgs_iters = 100
)

cat("\nChecking Pareto-K values from Pathfinder...\n")
print(pf())
