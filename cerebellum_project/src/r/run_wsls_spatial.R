library(cmdstanr)
library(dplyr)
library(readr)
library(yardstick)
library(loo)

dat <- read_rds('data/processed/behavioral_sample.rds')

# Spatial Target is Resp (1=Right, 2=Left)
dat_stan <- list(
  N_trials = nrow(dat),
  N_subj = length(unique(dat[['participant_id']])),
  subj = as.numeric(as.factor(dat[['participant_id']])),
  resp = dat[['Resp']],
  reward = dat[['reward']],
  rt = dat[['rt']],
  min_rt = dat %>% filter(rt > 0) %>% group_by(participant_id) %>% summarize(min_rt = min(rt, na.rm=TRUE)) %>% pull(min_rt),
  start_idx = dat %>% mutate(idx = row_number()) %>% group_by(participant_id) %>% summarize(idx = min(idx)) %>% pull(idx),
  end_idx = dat %>% mutate(idx = row_number()) %>% group_by(participant_id) %>% summarize(idx = max(idx)) %>% pull(idx),
  theta_mean = rep(0, 4),
  L_Sigma = diag(4),
  grainsize = 1
)

mod <- cmdstan_model('src/stan/wsls_spatial.stan', cpp_options = list(stan_threads = TRUE))

fit <- mod[['sample']](
  data = dat_stan,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 7,
  iter_warmup = 300,
  iter_sampling = 300,
  refresh = 10,
  init = 0,
  save_warmup = FALSE
)

fit[['save_object']]('results/fit_wsls_spatial.rds')
