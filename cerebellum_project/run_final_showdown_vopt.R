library(cmdstanr)
library(posterior)
library(dplyr)
library(loo)
library(pROC)
library(PRROC)
library(lme4)

set.seed(42)
options(mc.cores = 8)

cat("==========================================\n")
cat(" FINAL N=30 SHOWDOWN: M012 vs BASELINE V-OPT\n")
cat("==========================================\n\n")

# ------------------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------------------
cat("Loading N=30 Dataset...\n")
df_n30 <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N30.rds")
N_trials <- nrow(df_n30)
N_subj <- 30

W_exp <- matrix(rnorm(N_subj * 4, 0, 1), nrow=N_subj, ncol=4)
start_idx <- integer(N_subj)
end_idx <- integer(N_subj)
for(s in 1:N_subj) {
  start_idx[s] <- min(which(df_n30$subj_idx == s))
  end_idx[s] <- max(which(df_n30$subj_idx == s))
}

stan_data <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = df_n30$subj_idx,
  resp = df_n30$Boundary,
  reward = df_n30$F,
  rt = df_n30$RT,
  iti = df_n30$ITI,
  min_rt = df_n30 %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m),
  W_exp = W_exp,
  start_idx = start_idx,
  end_idx = end_idx,
  theta_mean = rep(0, 11),
  L_Sigma = diag(11),
  grainsize = 1
)

# ------------------------------------------------------------------------------
# 2. RUN BASELINE V-OPT
# ------------------------------------------------------------------------------
stan_data_vopt <- stan_data
stan_data_vopt$theta_mean <- rep(0, 7)
stan_data_vopt$L_Sigma <- diag(7)

cat("\nCompiling Baseline V-OPT...\n")
mod_vopt <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/Baseline_Optimized_GQ.stan", cpp_options = list(stan_threads = TRUE))

cat("Sampling Baseline V-OPT (N=30) ...\n")
fit_vopt <- mod_vopt$sample(
  data = stan_data_vopt,
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

# Save RAW output
fit_vopt$save_output_files(dir = "/home/DCCS5/cerebellum_project/results/", basename = "fit_vopt_n30")

# Diagnostics
summ_vopt <- fit_vopt$summary()
cat("Baseline V-OPT Diagnostics -> Max Rhat:", max(summ_vopt$rhat, na.rm=TRUE), "| Min Bulk ESS:", min(summ_vopt$ess_bulk, na.rm=TRUE), "\n")

# LOO
cat("Extracting LOO for Baseline V-OPT...\n")
loo_vopt <- fit_vopt$loo()
saveRDS(loo_vopt, "/home/DCCS5/cerebellum_project/results/loo_vopt_n30.rds")

# ------------------------------------------------------------------------------
# 3. RUN M012 (Control Run to match exact rng and inits)
# ------------------------------------------------------------------------------
cat("\nCompiling M012 Candidate...\n")
mod_m012 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/m012_candidate.stan", cpp_options = list(stan_threads = TRUE))

cat("Sampling M012 (N=30) ...\n")
fit_m012 <- mod_m012$sample(
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

fit_m012$save_output_files(dir = "/home/DCCS5/cerebellum_project/results/", basename = "fit_m012_ctrl_n30")

summ_m012 <- fit_m012$summary()
cat("M012 Diagnostics -> Max Rhat:", max(summ_m012$rhat, na.rm=TRUE), "| Min Bulk ESS:", min(summ_m012$ess_bulk, na.rm=TRUE), "\n")

cat("Extracting LOO for M012...\n")
loo_m012 <- fit_m012$loo()
saveRDS(loo_m012, "/home/DCCS5/cerebellum_project/results/loo_m012_ctrl_n30.rds")

# ------------------------------------------------------------------------------
# 4. FINAL LOO COMPARISON
# ------------------------------------------------------------------------------
cat("\n==========================================\n")
cat(" FINAL LOO COMPARISON\n")
cat("==========================================\n")
comp <- loo_compare(loo_m012, loo_vopt)
print(comp)

# Save the final text report
sink("/home/DCCS5/cerebellum_project/results/final_vopt_showdown.txt")
cat("FINAL LOO COMPARISON\n")
print(comp)
sink()

cat("\nDone! Pipeline complete.\n")
