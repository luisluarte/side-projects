local_lib <- Sys.getenv('R_LIBS_USER')
.libPaths(c(local_lib, .libPaths()))
if (!require('pacman')) install.packages('pacman', lib=local_lib, repos='http://cran.us.r-project.org')
pacman::p_load(cmdstanr, dplyr, readr, lme4, lmerTest, emmeans)

cat('LOADING DATA N=10...\n')
dat <- read_rds('data/processed/behavioral_sample.rds')
resp_clean <- dat %>% mutate(Resp = case_when(Resp %in% c(1,2) ~ Resp, TRUE ~ -999)) %>% pull(Resp)

dat_stan <- list(
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
  theta_mean = rep(0, 12),
  L_Sigma = diag(12),
  grainsize = 1
)

dat_stan_wsls <- dat_stan
dat_stan_wsls[['theta_mean']] <- rep(0, 4)
dat_stan_wsls[['L_Sigma']] <- diag(4)

cat('COMPILING MODELS...\n')
mod_m012 <- cmdstan_model('src/stan/m012_spatial.stan', cpp_options = list(stan_threads = TRUE))
mod_wsls <- cmdstan_model('src/stan/wsls_spatial.stan', cpp_options = list(stan_threads = TRUE))

cat('FITTING WSLS N=10...\n')
fit_wsls <- mod_wsls[['sample']](data = dat_stan_wsls, chains = 4, parallel_chains = 4, threads_per_chain = 7, iter_warmup = 300, iter_sampling = 300, refresh = 0, init = 0)

cat('FITTING M012 N=10...\n')
fit_m012 <- mod_m012[['sample']](data = dat_stan, chains = 4, parallel_chains = 4, threads_per_chain = 7, iter_warmup = 300, iter_sampling = 300, refresh = 0, init = 0)

cat('EXTRACTING LOG-LIKELIHOODS...\n')
ll_wsls <- fit_wsls[['summary']]('log_lik', 'median')[['median']]
ll_m012 <- fit_m012[['summary']]('log_lik', 'median')[['median']]

dat_wsls <- dat %>% mutate(log_lik = ll_wsls, model = 'WSLS')
dat_m012 <- dat %>% mutate(log_lik = ll_m012, model = 'M012')
dat_combined <- bind_rows(dat_wsls, dat_m012) %>% filter(Resp %in% c(1, 2) & rt > 0)

cat('FITTING LMM...\n')
m <- lmer(log_lik ~ nt * model + (1 | participant_id), data = dat_combined)

cat('\n--- EMTRENDS RESULTS ---\n')
emt <- emtrends(m, pairwise ~ model, var = 'nt')
print(emt)

