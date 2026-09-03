library(cmdstanr)
library(dplyr)
library(readr)

dat <- read_rds('data/processed/behavioral_compilate.rds')
resp_clean <- dat %>% mutate(Resp = case_when(Resp %in% c(1,2) ~ Resp, TRUE ~ -999)) %>% pull(Resp)

dat_stan_vopt <- list(
  N_trials = nrow(dat),
  N_subj = length(unique(dat[['participant_id']])),
  subj = as.numeric(as.factor(dat[['participant_id']])),
  resp = resp_clean,
  reward = dat[['reward']],
  rt = dat[['rt']],
  iti = dat[['ttF']],
  W_exp = matrix(rnorm(length(unique(dat[['participant_id']])) * 32), ncol=32),
  min_rt = dat %>% filter(rt > 0) %>% group_by(participant_id) %>% summarize(min_rt = min(rt, na.rm=TRUE)) %>% pull(min_rt),
  start_idx = dat %>% mutate(idx = row_number()) %>% group_by(participant_id) %>% summarize(idx = min(idx)) %>% pull(idx),
  end_idx = dat %>% mutate(idx = row_number()) %>% group_by(participant_id) %>% summarize(idx = max(idx)) %>% pull(idx),
  theta_mean = rep(0, 8),
  L_Sigma = diag(8),
  grainsize = 1
)

dat_stan_m012 <- dat_stan_vopt
dat_stan_m012[['theta_mean']] <- rep(0, 12)
dat_stan_m012[['L_Sigma']] <- diag(12)

cat('COMPILING VOPT SPATIAL\n')
mod_vopt <- cmdstan_model('src/stan/vopt_spatial.stan', cpp_options = list(stan_threads = TRUE))
cat('COMPILING M012 SPATIAL\n')
mod_m012 <- cmdstan_model('src/stan/m012_spatial.stan', cpp_options = list(stan_threads = TRUE))

cat('FITTING VOPT SPATIAL N=30\n')
fit_vopt <- mod_vopt[['sample']](
  data = dat_stan_vopt,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 7,
  iter_warmup = 600,
  iter_sampling = 600,
  refresh = 10,
  init = 0,
  save_warmup = FALSE
)
fit_vopt[['save_object']]('results/fit_vopt_spatial_n30.rds')

cat('FITTING M012 SPATIAL N=30\n')
fit_m012 <- mod_m012[['sample']](
  data = dat_stan_m012,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 7,
  iter_warmup = 600,
  iter_sampling = 600,
  refresh = 10,
  init = 0,
  save_warmup = FALSE
)
fit_m012[['save_object']]('results/fit_m012_spatial_n30.rds')
