library(cmdstanr)
library(dplyr)
library(readr)

cat("Initializing CmdStanR (Phase III: Empirical Literature Priors)...\n")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=FALSE)
epistemic <- readRDS("results/epistemic_geometry.rds")

test_dat <- dat_raw %>% 
    filter(participant_id %in% epistemic$test_subjs) %>%
    arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(F)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), 
           subj_idx = as.integer(as.factor(participant_id)))

subj_indices <- test_dat %>%
    mutate(row_num = row_number()) %>%
    group_by(subj_idx) %>%
    summarise(start_idx = min(row_num), end_idx = max(row_num)) %>%
    arrange(subj_idx)

min_rt_df <- test_dat %>% group_by(subj_idx) %>% summarise(min_rt = min(RT)) %>% arrange(subj_idx)

stan_data <- list(
    N_trials = nrow(test_dat),
    N_subj = max(test_dat$subj_idx),
    subj = test_dat$subj_idx,
    resp = test_dat$Boundary,
    reward = test_dat$F,
    rt = test_dat$RT,
    iti = test_dat$ITI,
    min_rt = min_rt_df$min_rt,
    W_exp = matrix(rnorm(max(test_dat$subj_idx) * 32, 0, 1), nrow=max(test_dat$subj_idx), ncol=32),
    start_idx = subj_indices$start_idx,
    end_idx = subj_indices$end_idx
)

cat("Compiling Stan model with Tran et al. (2021) Priors...\n")
mod <- cmdstan_model("src/stan/m006_strict_hmc.stan")

cat("Starting HMC sampling...\n")
fit <- mod$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 100,      
    iter_sampling = 100,    
    adapt_engaged = TRUE,
    step_size = 0.05 
)

fit$save_object("results/hmc_phase3_fit.rds")
print(fit$summary())
