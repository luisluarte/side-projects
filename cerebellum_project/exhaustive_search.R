pacman::p_load(tidyverse, lme4, lmerTest)
sim_data <- read_csv("results/tables/magi_raw_trial_simulations_N30.csv", show_col_types = FALSE)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)

dat <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id))) %>% filter(participant_idx <= 30)

dat <- dat %>% group_by(participant_idx) %>% mutate(Cond_Resp = as.character(Resp)) %>% ungroup()
sim_data$Cond_Resp <- dat$Cond_Resp
df <- sim_data %>% filter(!is.na(Cond_Resp))

cat("\n=== CONDITION: SPATIAL CHOICE (Resp 2 vs Resp 1) ===\n")
mod <- lmer(Empirical_RT ~ Cond_Resp + (1|SubjectID), data=df)
print(summary(mod)$coefficients)

subj_deltas <- df %>% group_by(SubjectID, Cond_Resp) %>%
    summarize(Emp=mean(Empirical_RT), Base=mean(Baseline_Wald_Sim_RT), Term=mean(Terminal_Hybrid_Sim_RT), .groups="drop") %>%
    pivot_wider(names_from = Cond_Resp, values_from = c(Emp, Base, Term))

cat("\n--- Delta RT (Resp 2 vs Resp 1) ---\n")
cat("Empirical Delta: ", mean(subj_deltas$`Emp_2` - subj_deltas$`Emp_1`, na.rm=TRUE)*1000, "ms\n")
cat("Baseline Delta:  ", mean(subj_deltas$`Base_2` - subj_deltas$`Base_1`, na.rm=TRUE)*1000, "ms\n")
cat("Terminal Delta:  ", mean(subj_deltas$`Term_2` - subj_deltas$`Term_1`, na.rm=TRUE)*1000, "ms\n")
