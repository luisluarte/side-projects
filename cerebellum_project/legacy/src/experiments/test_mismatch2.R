
pacman::p_load(tidyverse, cmdstanr, posterior)
fit_q <- readRDS('../../results/fit_q_complete.rds')

dat_raw <- read_csv('../../data/raw/behavioral_compilate.csv', show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(F))
subject_counts <- dat_clean %>% group_by(participant_id) %>% summarise(min_rt = min(RT)) %>% ungroup()
# Since we didn't save min_rt for Q, just take the first 30 subjects
found_pids <- unique(dat_clean[['participant_id']])[1:30]
dat_reconstructed <- dat_clean %>% filter(participant_id %in% found_pids) %>% group_by(participant_id) %>% mutate(ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup()
rescued_counts <- dat_reconstructed %>% group_by(participant_id) %>% summarise(count = n(), min_rt = min(RT)) %>% ungroup()
end_idx <- cumsum(rescued_counts[['count']])
start_idx <- end_idx - rescued_counts[['count']] + 1
stan_data <- list(N = nrow(dat_reconstructed), S = nrow(rescued_counts), start_idx = start_idx, end_idx = end_idx, choice = dat_reconstructed[['Boundary']], rt = dat_reconstructed[['RT']], reward = dat_reconstructed[['F']], iti = dat_reconstructed[['ITI']], f_dur = dat_reconstructed[['F_dur']], min_rt = rescued_counts[['min_rt']], N_MF = 5, grainsize = 1)

mod_q_gq <- cmdstan_model('../models/q_learning_ddm_gq.stan')
gq_q <- mod_q_gq[['generate_quantities']](fitted_params = fit_q, data = stan_data, parallel_chains = 4)

