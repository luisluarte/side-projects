# setup -------------------------------------------------------------------
local_lib <- Sys.getenv("R_LIBS_USER")

if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

# libs --------------------------------------------------------------------
cat("LIBS\n")
if (!require("pacman", character.only = TRUE)) {
  install.packages("pacman",
    lib = local_lib
  )
  library("pacman", character.only = TRUE)
}
pacman::p_load(
  tidyverse,
  cmdstanr,
  posterior,
  this.path
)

if (!dir.exists(cmdstan_path())) {
  install_cmdstan()
} else {
  message("cmdstan installed at: ", cmdstan_path())
}

setwd(here())

# data --------------------------------------------------------------------

cat("DATA PROC\n")
dat_raw <- read_csv("../../data/raw/behavioral_compilate.csv") %>%
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
write_rds(x = dat_raw, file = "../../data/processed/behavioral_compilate.rds")

# dat <- dat_raw %>%
#   filter(participant_id %in% sample(
#     x = unique(dat_raw$participant_id),
#     size = 10,
#     replace = FALSE
#   ))
# write_rds(x = dat, file = "../../data/processed/behavioral_sample.rds")

N_trials <- nrow(dat)
N_subj <- length(unique(dat %>% pull(participant_id)))
subj <- as.numeric(
  as.factor(
    dat %>% pull(participant_id)
  )
)
# tay = 1; switch = 2
resp <- dat %>%
  pull(stay_switch) %>%
  {
    ifelse(. == "stay", 1, 2)
  }
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
L_Sigma_m012 <- diag(12)
# mini expansion couse task too easy
W_exp <- matrix(0, nrow = N_subj, ncol = 4)

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
cat("COMPILING VOPT\n")
mod_vopt <- cmdstan_model(
  "../stan/vopt_ss3.stan",
  cpp_options = list(stan_threads = TRUE)
)
cat("COMPILING M012\n")
mod_m012 <- cmdstan_model(
  "../stan/m012_ss3.stan",
  cpp_options = list(stan_threads = TRUE)
)

# fit ---------------------------------------------------------------------
cat("FIT VOPT\n")
fit_vopt <- mod_vopt$sample(
  data = stan_data_vopt,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 8,
  iter_warmup = 300,
  iter_sampling = 300,
  refresh = 10,
  init = 0
)
cat("FIT M012\n")
fit_m012 <- mod_m012$sample(
  data = stan_data_m012,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 8,
  iter_warmup = 300,
  iter_sampling = 300,
  refresh = 10,
  init = 0
)
cat("SAVE MODELS\n")
fit_vopt$save_object(file = "../../results/fit_vopt.rds")
fit_m012$save_object(file = "../../results/fit_m012.rds")
