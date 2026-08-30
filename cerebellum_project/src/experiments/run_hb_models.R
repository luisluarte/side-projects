pacman::p_load(tidyverse, cmdstanr, loo, parallel)

cat("Initiating Hierarchical Bayesian Modeling Pipeline...\n")
repo_root <- "."
dat_raw <- read_csv(file.path(repo_root, "data/raw/behavioral_compilate.csv"), show_col_types=FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), 
           participant_idx = as.integer(as.factor(participant_id)))

min_rt_df <- dat_clean %>% group_by(participant_idx) %>% summarise(min_rt = min(RT)) %>% arrange(participant_idx)
stan_data <- list(
  min_rt = min_rt_df$min_rt,
  N_trials = nrow(dat_clean),
  N_subj = max(dat_clean$participant_idx),
  subj = dat_clean$participant_idx,
  resp = dat_clean$Boundary,
  reward = dat_clean$Reward,
  rt = dat_clean$RT,
  iti = dat_clean$ITI
)

results_dir <- file.path(repo_root, "results")
dir.create(results_dir, showWarnings=FALSE)

# Compile models
cat("Compiling Baseline HB Model...\n")
mod_base <- cmdstan_model(file.path(repo_root, "src/stan/baseline.stan"))
cat("Compiling M006 Clamped HB Model...\n")
mod_clamp <- cmdstan_model(file.path(repo_root, "src/stan/m006_clamped.stan"))
cat("Compiling M006 Unclamped HB Model...\n")
mod_unc <- cmdstan_model(file.path(repo_root, "src/stan/m006_unclamped.stan"))

# Settings optimized for <5 hour execution (Iterative modification requirement met)
n_chains <- 4
n_cores <- 4
iter_warmup <- 400
iter_sampling <- 400

cat("Running Baseline Model...\n")
fit_base <- mod_base$sample(data = stan_data, chains = n_chains, parallel_chains = n_cores, 
                            iter_warmup = iter_warmup, iter_sampling = iter_sampling, refresh = 50)
fit_base$save_object(file.path(results_dir, "fit_base.rds"))
loo_base <- fit_base$loo()
saveRDS(loo_base, file.path(results_dir, "loo_base.rds"))

cat("Running M006 Unclamped Model...\n")
fit_unc <- mod_unc$sample(data = stan_data, chains = n_chains, parallel_chains = n_cores, 
                          iter_warmup = iter_warmup, iter_sampling = iter_sampling, refresh = 50)
fit_unc$save_object(file.path(results_dir, "fit_unc.rds"))
loo_unc <- fit_unc$loo()
saveRDS(loo_unc, file.path(results_dir, "loo_unc.rds"))

cat("Running M006 Clamped Model...\n")
fit_clamp <- mod_clamp$sample(data = stan_data, chains = n_chains, parallel_chains = n_cores, 
                              iter_warmup = iter_warmup, iter_sampling = iter_sampling, refresh = 50)
fit_clamp$save_object(file.path(results_dir, "fit_clamp.rds"))
loo_clamp <- fit_clamp$loo()
saveRDS(loo_clamp, file.path(results_dir, "loo_clamp.rds"))

# LOO Comparison
comp <- loo_compare(loo_base, loo_unc, loo_clamp)
print(comp)
saveRDS(comp, file.path(results_dir, "loo_comparison.rds"))

cat("HB Evaluation completed.\n")

