library(cmdstanr)
library(loo)
library(dplyr)
library(readr)

cat("========================================================\n")
cat("      PHASE V: THERMODYNAMIC LOO-CV COMPARISON\n")
cat("========================================================\n")

cat("Loading and processing models...\n")
fit_base <- readRDS("results/hmc_baseline_fit.rds")
loo_base <- fit_base$loo(cores = 4)

fit_m006 <- readRDS("results/hmc_phase3_fit.rds")

# Re-run GQ if needed or just load if it was saved? We didn't save it. We'll run it again.
mod_gq <- cmdstan_model("src/stan/m006_gq.stan")
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=FALSE)
epistemic <- readRDS("results/epistemic_geometry.rds")

test_dat <- dat_raw %>% filter(participant_id %in% epistemic$test_subjs) %>%
    arrange(participant_id, ttp) %>% group_by(participant_id) %>%
    mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(F)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), subj_idx = as.integer(as.factor(participant_id)))

subj_indices <- test_dat %>% mutate(row_num = row_number()) %>% group_by(subj_idx) %>% summarise(start_idx = min(row_num), end_idx = max(row_num)) %>% arrange(subj_idx)
min_rt_df <- test_dat %>% group_by(subj_idx) %>% summarise(min_rt = min(RT)) %>% arrange(subj_idx)

stan_data <- list(N_trials = nrow(test_dat), N_subj = max(test_dat$subj_idx), subj = test_dat$subj_idx, resp = test_dat$Boundary, reward = test_dat$F, rt = test_dat$RT, iti = test_dat$ITI, min_rt = min_rt_df$min_rt, W_exp = matrix(rnorm(max(test_dat$subj_idx)*32,0,1),nrow=max(test_dat$subj_idx),ncol=32), start_idx = subj_indices$start_idx, end_idx = subj_indices$end_idx, theta_mean = epistemic$theta_train_mean, L_Sigma = t(chol(epistemic$Sigma_train)))

fit_m006_gq <- mod_gq$generate_quantities(fitted_params = fit_m006, data = stan_data, parallel_chains = 4)

cat("Extracting log_lik and computing LOO...\n")
log_lik_m006 <- fit_m006_gq$draws("log_lik")
loo_m006 <- loo::loo(log_lik_m006, cores = 4)

cat("========================================================\n")
cat("      FINAL LOO-CV ELPD COMPARISON RESULTS\n")
cat("========================================================\n")
comp <- loo_compare(loo_base, loo_m006)
print(comp)

saveRDS(list(loo_base=loo_base, loo_m006=loo_m006, comp=comp), "results/final_loo_comparison.rds")
