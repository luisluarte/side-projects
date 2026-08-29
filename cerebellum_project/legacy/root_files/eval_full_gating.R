pacman::p_load(tidyverse, cmdstanr, posterior, loo)

dat_raw <- read_csv("~/cerebellum_project/data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
set.seed(420)
pid_sample <- sample(unique(dat_clean$participant_id), size = 30)
dat_clean <- dat_clean %>% filter(participant_id %in% pid_sample)
subject_counts <- dat_clean %>% group_by(participant_id) %>% summarise(count = n()) %>% mutate(end_idx = cumsum(count), start_idx = end_idx - count + 1)
min_rt_df <- dat_clean %>% group_by(participant_id) %>% summarise(min_rt = min(RT))
stan_data <- list(N = nrow(dat_clean), S = nrow(subject_counts), start_idx = subject_counts$start_idx, end_idx = subject_counts$end_idx, choice = dat_clean$Boundary, rt = dat_clean$RT, reward = dat_clean$`F`, iti = dat_clean$ITI, f_dur = dat_clean$F_dur, min_rt = min_rt_df$min_rt, N_MF = 5, grainsize = 1)

fit_gating <- read_rds("~/cerebellum_project/results/fit_full_gating_complete.rds")
mod_gating_gq <- cmdstan_model("~/cerebellum_project/src/models/bvk_full_gating_gq.stan")

gq_gating <- mod_gating_gq$generate_quantities(fitted_params = fit_gating, data = stan_data, parallel_chains = 4)
loo_gating <- gq_gating$loo()

cat("LOO Full Gating:\n")
print(loo_gating)

# We can also compare against Q-learning if we have fit_q_complete.rds
fit_q <- read_rds("~/cerebellum_project/results/fit_q_complete.rds")
mod_q_gq <- cmdstan_model("~/cerebellum_project/src/models/q_learning_ddm_gq.stan")
gq_q <- mod_q_gq$generate_quantities(fitted_params = fit_q, data = stan_data, parallel_chains = 4)
loo_q <- gq_q$loo()

cat("LOO Compare (Gating vs Q):\n")
print(loo_compare(loo_q, loo_gating))

write_rds(loo_gating, "~/cerebellum_project/results/loo_full_gating.rds")
