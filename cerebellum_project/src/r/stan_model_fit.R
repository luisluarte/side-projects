# libs --------------------------------------------------------------------
pacman::p_load(
  tidyverse,
  cmdstanr,
  posterior,
  this.path
)

setwd(here())

# data --------------------------------------------------------------------

dat <- read_csv("../../data/raw/behavioral_compilate.csv") %>%
  group_by(participant_id) %>%
  mutate(
    stay_switch = if_else(
      Resp == lag(Resp, n = 1),
      "stay",
      "switch"
    ),
    stay_switch = replace_na(stay_switch, "stay"),
    reward = as.numeric(`F`),
    rt_raw = (ttr - ttp) / 1000,
    rt = ifelse(rt_raw < 0.15 | rt_raw > 5.0, -999, rt_raw),
    iti = replace_na(ttp - lag(ttF, n = 1), 0)
  ) %>%
  ungroup()

N_trials <- nrow(dat)
N_subj <- length(unique(dat %>% pull(participant_id)))
subj <- as.numeric(
  as.factor(
    dat %>% pull(participant_id)
  ))
#tay = 1; switch = 2
resp <- dat %>%
  pull(stay_switch) %>%
  {ifelse(. == "stay", 1, 2)}
# reward
reward <- dat %>% pull(reward)
rt <- dat %>% pull(rt)
iti <- dat %>% pull(iti)
min_rt <- dat %>%
  filter(rt > 0) %>%
  group_by(participant_id) %>%
  summarise(min_rt = min(rt, na.rm = TRUE)) %>%
  pull(min_rt)
start_idx <- dat %>%
  mutate(r = row_number()) %>%
  filter(nt == 1) %>%
  select(participant_id, r) %>%
  pull(r)
end_idx <- dat %>%
  mutate(r = row_number()) %>%
  group_by(participant_id) %>%
  slice_max(order_by = nt, n = 1) %>%
  select(participant_id, r) %>%
  pull(r)
theta_mean_vopt <- rep(0, 8)
theta_mean_m012 <- rep(0, 12)
# identity while I find a way to inform it
L_Sigma_vopt <- diag(8)
L_Sigma_m012 <- diag(8)
# mini expansion couse task too easy
W_exp <- matrix(0, nrow=N_subj, ncol=4)

stan_data_vopt <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = subj,
  resp = resp,
  reward = reward,
  rt = rt,
  min_rt = min_rt,
  start_idx = start_idx,
  end_idx = end_idx,
  theta_mean = theta_mean_vopt,
  L_Sigma = L_Sigma_vopt,
  grainsize = 1
)

stan_data_m012 <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = subj,
  resp = resp,
  reward = reward,
  rt = rt,
  iti = iti,
  min_rt = min_rt,
  start_idx = start_idx,
  end_idx = end_idx,
  W_exp = W_exp,
  theta_mean = theta_mean_m012,
  L_Sigma = L_Sigma_m012,
  grainsize = 1
)


# stan models -------------------------------------------------------------
mod_vopt <- cmdstan_model(
  "../stan/vopt_ss3.stan",
  cpp_options = list(stan_threads = TRUE))
mod_m012 <- cmdstan_model(
  "../stan/m012_ss3.stan",
  cpp_options = list(stan_threads = TRUE))

# fit ---------------------------------------------------------------------
fit_vopt <- mod_vopt$sample(
  data = stan_data_vopt,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  iter_warmup = 300,
  iter_sampling = 300,
  refresh = 100,
  init = 0
  )
fit_m012 <- mod_m012$sample(
  data = stan_data_m012,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  iter_warmup = 300,
  iter_sampling = 300,
  refresh = 100,
  init = 0
  )
