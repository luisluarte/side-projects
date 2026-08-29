pacman::p_load(tidyverse, lme4, lmerTest)
sim_data <- read_csv("results/tables/magi_raw_trial_simulations_N30.csv", show_col_types = FALSE)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)

dat <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id))) %>% filter(participant_idx <= 30)

dat <- dat %>% group_by(participant_idx) %>%
    mutate(
        Prev_RT = lag(RT),
        q25 = quantile(RT, 0.25, na.rm=TRUE),
        q75 = quantile(RT, 0.75, na.rm=TRUE),
        Cond_PrevRT = case_when(Prev_RT <= q25 ~ "1_PrevFast", Prev_RT >= q75 ~ "2_PrevSlow", TRUE ~ NA_character_)
    ) %>% ungroup()

sim_data$Cond_PrevRT <- dat$Cond_PrevRT
df <- sim_data %>% filter(!is.na(Cond_PrevRT))

cat("\n=== CONDITION: AUTOCORRELATION (Trial after FAST vs Trial after SLOW) ===\n")
mod <- lmer(Empirical_RT ~ Cond_PrevRT + (1|SubjectID), data=df)
print(summary(mod)$coefficients)

subj_deltas <- df %>% group_by(SubjectID, Cond_PrevRT) %>%
    summarize(Emp=mean(Empirical_RT), Base=mean(Baseline_Wald_Sim_RT), Term=mean(Terminal_Hybrid_Sim_RT), .groups="drop") %>%
    pivot_wider(names_from = Cond_PrevRT, values_from = c(Emp, Base, Term))
    
cat("\n--- Delta RT (Post-Slow vs Post-Fast) ---\n")
cat("Empirical Delta: ", mean(subj_deltas$`Emp_2_PrevSlow` - subj_deltas$`Emp_1_PrevFast`, na.rm=TRUE)*1000, "ms\n")
cat("Baseline Delta:  ", mean(subj_deltas$`Base_2_PrevSlow` - subj_deltas$`Base_1_PrevFast`, na.rm=TRUE)*1000, "ms\n")
cat("Terminal Delta:  ", mean(subj_deltas$`Term_2_PrevSlow` - subj_deltas$`Term_1_PrevFast`, na.rm=TRUE)*1000, "ms\n")
