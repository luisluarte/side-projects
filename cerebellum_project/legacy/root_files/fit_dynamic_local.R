pacman::p_load(tidyverse, cmdstanr)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

set.seed(420)
pid_sample <- sample(unique(dat_clean$participant_id), size = 3) # small sample
dat_clean <- dat_clean %>% filter(participant_id %in% pid_sample) %>% mutate(participant_idx = as.integer(as.factor(participant_id)))

start_idx <- dat_clean %>% group_by(participant_idx) %>% summarise(start = min(row_number()), .groups="drop") %>% pull(start)
end_idx <- dat_clean %>% group_by(participant_idx) %>% summarise(end = max(row_number()), .groups="drop") %>% pull(end)
min_rts <- dat_clean %>% group_by(participant_idx) %>% summarise(min_rt = min(RT), .groups="drop") %>% pull(min_rt)

stan_data <- list(
  N = nrow(dat_clean), S = length(unique(dat_clean$participant_idx)),
  start_idx = start_idx, end_idx = end_idx,
  choice = dat_clean$Boundary, rt = dat_clean$RT, reward = dat_clean$`F`, iti = dat_clean$ITI,
  min_rt = min_rts, grainsize = 1
)

mod <- cmdstan_model("src/models/q_learning_ddm_dynamic.stan")
fit <- mod$sample(data = stan_data, chains = 2, parallel_chains = 2, iter_warmup = 300, iter_sampling = 300, adapt_delta = 0.95, init = 0.1)
fit$summary(c("mu_alpha_ctx", "mu_kappa_ctx", "mu_a", "mu_tau_nd", "mu_theta_ctx", "mu_beta_a"))

mod_gq <- cmdstan_model("src/models/q_learning_ddm_dynamic_gq.stan")
gq <- mod_gq$generate_quantities(fitted_params = fit, data = stan_data, parallel_chains = 2)
ll <- gq$draws("log_lik")
library(loo)
print(loo(ll))
