library(cmdstanr)
library(dplyr)
library(readr)

dat_raw <- read_csv("/home/DCCS5/cerebellum_project/data/raw/behavioral_compilate.csv", show_col_types=FALSE)
# Get subjects 11 to 20
test_subjs_10 <- unique(dat_raw$participant_id)[11:20]
test_dat <- dat_raw %>% filter(participant_id %in% test_subjs_10) %>%
    arrange(participant_id, ttp) %>% group_by(participant_id) %>%
    mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(F)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), subj_idx = as.integer(as.factor(participant_id)))

# Save dat
saveRDS(test_dat, "/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N10.rds")

stan_data <- list(
  N_trials = nrow(test_dat),
  N_subj = max(test_dat$subj_idx),
  subj = test_dat$subj_idx,
  resp = test_dat$Boundary,
  reward = test_dat$F,
  rt = test_dat$RT,
  iti = test_dat$ITI
)

# Extract subject bounds
min_rt <- test_dat %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m)
stan_data$min_rt <- min_rt

# Start and end indices
start_idx <- numeric(stan_data$N_subj)
end_idx <- numeric(stan_data$N_subj)
for(s in 1:stan_data$N_subj) {
  idx <- which(test_dat$subj_idx == s)
  start_idx[s] <- min(idx)
  end_idx[s] <- max(idx)
}
stan_data$start_idx <- start_idx
stan_data$end_idx <- end_idx

# W_exp
set.seed(42)
W_exp <- matrix(rnorm(stan_data$N_subj * 32, 0, 1), nrow=stan_data$N_subj, ncol=32)
stan_data$W_exp <- W_exp

# Prior Preconditioning (From CMA-ES)
stan_data$theta_mean <- c(1.36, -0.63, 0.44, -1.94, -0.06, -0.09, 1.48, 1.05, -0.58)
cov_matrix <- matrix(c(
  0.13, 0.04, 0.01, 0.03, 0.00, 0.04, -0.01, -0.02, 0.01,
  0.04, 0.17, 0.02, 0.05, 0.01, 0.02, 0.01, -0.03, -0.02,
  0.01, 0.02, 0.22, 0.01, 0.02, 0.03, 0.05, 0.02, 0.01,
  0.03, 0.05, 0.01, 0.11, -0.02, -0.01, 0.03, -0.01, 0.00,
  0.00, 0.01, 0.02, -0.02, 0.08, 0.04, 0.02, -0.01, 0.01,
  0.04, 0.02, 0.03, -0.01, 0.04, 0.15, 0.04, 0.03, -0.02,
  -0.01, 0.01, 0.05, 0.03, 0.02, 0.04, 0.14, -0.01, 0.02,
  -0.02, -0.03, 0.02, -0.01, -0.01, 0.03, -0.01, 0.12, 0.00,
  0.01, -0.02, 0.01, 0.00, 0.01, -0.02, 0.02, 0.00, 0.09
), nrow=9, byrow=TRUE)
stan_data$L_Sigma <- t(chol(cov_matrix))

# Compile Baseline
cat("Compiling Baseline...\n")
mod_base <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/baseline.stan")
cat("Fitting Baseline...\n")
fit_base <- mod_base$sample(data = stan_data, chains = 4, parallel_chains = 4, iter_warmup = 300, iter_sampling = 300, init = 0, adapt_delta = 0.95, step_size = 0.05, max_treedepth = 10)
fit_base$save_object("/home/DCCS5/cerebellum_project/results/baseline_urgency.rds")

# Compile Urgency M007
cat("Compiling Urgency M007...\n")
mod_urgency <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/m007_urgency.stan")
cat("Fitting Urgency M007...\n")
fit_urgency <- mod_urgency$sample(data = stan_data, chains = 4, parallel_chains = 4, iter_warmup = 300, iter_sampling = 300, init = 0, adapt_delta = 0.95, step_size = 0.05, max_treedepth = 10)
fit_urgency$save_object("/home/DCCS5/cerebellum_project/results/m007_urgency.rds")
cat("DONE\n")