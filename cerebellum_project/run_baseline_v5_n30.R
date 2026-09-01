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
cat(" RUNNING BASELINE V5 (N=30) AND FINAL METRICS\n")
cat("==========================================\n\n")

df_n30 <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N30.rds")
N_trials <- nrow(df_n30)
N_subj <- 30

# Calculate rolling EMA entropy per subject
calc_rolling_entropy <- function(rewards, alpha = 0.5) {
  n <- length(rewards)
  ent <- numeric(n)
  p <- 0.5 # start at maximum uncertainty
  for (i in 1:n) {
    p_clip <- min(max(p, 0.01), 0.99)
    ent[i] <- - (p_clip * log2(p_clip) + (1 - p_clip) * log2(1 - p_clip))
    p <- alpha * rewards[i] + (1 - alpha) * p
  }
  return(ent)
}

df_n30 <- df_n30 %>%
  group_by(subj_idx) %>%
  mutate(roll_ent = calc_rolling_entropy(F, alpha = 0.5)) %>%
  ungroup()

start_idx <- integer(N_subj)
end_idx <- integer(N_subj)
for(s in 1:N_subj) {
  start_idx[s] <- min(which(df_n30$subj_idx == s))
  end_idx[s] <- max(which(df_n30$subj_idx == s))
}

stan_data_v5 <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = df_n30$subj_idx,
  resp = df_n30$Boundary,
  reward = df_n30$F,
  rt = df_n30$RT,
  iti = df_n30$ITI,
  entropy = df_n30$roll_ent,
  min_rt = df_n30 %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m),
  start_idx = start_idx,
  end_idx = end_idx,
  grainsize = 1
)

cat("Compiling Baseline V5...\n")
mod_v5 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/baseline_v5.stan", cpp_options = list(stan_threads = TRUE))

cat("Sampling Baseline V5 (N=30) ...\n")
fit_v5 <- mod_v5$sample(
  data = stan_data_v5,
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
fit_v5$save_output_files(dir = "/home/DCCS5/cerebellum_project/results/", basename = "fit_v5_n30")

summ_v5 <- fit_v5$summary()
cat("Baseline V5 Diagnostics -> Max Rhat:", max(summ_v5$rhat, na.rm=TRUE), "| Min Bulk ESS:", min(summ_v5$ess_bulk, na.rm=TRUE), "\n")
loo_v5 <- fit_v5$loo()
saveRDS(loo_v5, "/home/DCCS5/cerebellum_project/results/loo_v5_n30.rds")

loo_m012 <- readRDS("/home/DCCS5/cerebellum_project/results/loo_m012_n30.rds")
cat("\n==========================================\n")
cat(" LOO-CV COMPARISON (N=30)\n")
cat("==========================================\n")
comp <- loo_compare(list(M012 = loo_m012, BaselineV5 = loo_v5))
print(comp)
cat("\n")
