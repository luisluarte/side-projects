pacman::p_load(tidyverse, cmdstanr, posterior)

dat_raw <- read_csv("~/cerebellum_project/data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
set.seed(420)
pid_sample <- sample(unique(dat_clean$participant_id), size = 30)
dat_clean <- dat_clean %>% filter(participant_id %in% pid_sample)
subject_counts <- dat_clean %>% group_by(participant_id) %>% summarise(count = n()) %>% mutate(end_idx = cumsum(count), start_idx = end_idx - count + 1)
min_rt_df <- dat_clean %>% group_by(participant_id) %>% summarise(min_rt = min(RT))
stan_data <- list(N = nrow(dat_clean), S = nrow(subject_counts), start_idx = subject_counts$start_idx, end_idx = subject_counts$end_idx, choice = dat_clean$Boundary, rt = dat_clean$RT, reward = dat_clean$`F`, iti = dat_clean$ITI, f_dur = dat_clean$F_dur, min_rt = min_rt_df$min_rt, N_MF = 5, grainsize = 1)

mod_gating <- cmdstan_model("~/cerebellum_project/src/models/bvk_full_gating.stan", cpp_options = list(stan_threads = TRUE))

cat("Fitting Full Gating N=30...\n")
fit_gating <- mod_gating$sample(data = stan_data, chains = 4, parallel_chains = 4, threads_per_chain = 8, iter_warmup = 500, iter_sampling = 500, adapt_delta = 0.95, max_treedepth = 12)

fit_gating$save_object("~/cerebellum_project/results/fit_full_gating_complete.rds")
