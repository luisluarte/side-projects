pacman::p_load(tidyverse, lme4, lmerTest)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)

dat <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id))) %>% filter(participant_idx <= 30)

dat <- dat %>% group_by(participant_idx) %>% mutate(
    Is_Switch = ifelse(row_number() > 1 & Resp != lag(Resp), "2_Switch", "1_Stay"),
    Choice_Run = sequence(rle(Resp)$lengths),
    Run_Cond = case_when(Choice_Run == 1 ~ "1_Switch", Choice_Run >= 4 ~ "2_DeepStay", TRUE ~ NA_character_)
) %>% ungroup()

cat("\n=== EMPIRICAL LMER: SWITCH COST ===\n")
mod_sw <- lmer(RT ~ Is_Switch + (1|participant_idx), data=dat %>% filter(!is.na(Is_Switch)))
print(summary(mod_sw)$coefficients)

cat("\n=== EMPIRICAL LMER: DEEP STAY VS SWITCH ===\n")
mod_run <- lmer(RT ~ Run_Cond + (1|participant_idx), data=dat %>% filter(!is.na(Run_Cond)))
print(summary(mod_run)$coefficients)

subj_deltas <- dat %>% group_by(participant_idx, Run_Cond) %>%
    filter(!is.na(Run_Cond)) %>%
    summarize(Emp=mean(RT), .groups="drop") %>%
    pivot_wider(names_from = Run_Cond, values_from = c(Emp))
    
cat("\nEmpirical Delta (DeepStay - Switch): ", mean(subj_deltas$`2_DeepStay` - subj_deltas$`1_Switch`, na.rm=TRUE)*1000, "ms\n")

sim_data <- read_csv("results/tables/magi_raw_trial_simulations_N30.csv", show_col_types = FALSE)
sim_data$Run_Cond <- dat$Run_Cond
df <- sim_data %>% filter(!is.na(Run_Cond))

cat("\n--- Delta RT (DeepStay - Switch) ---\n")
subj_deltas_sim <- df %>% group_by(SubjectID, Run_Cond) %>%
    summarize(Base=mean(Baseline_Wald_Sim_RT), Term=mean(Terminal_Hybrid_Sim_RT), .groups="drop") %>%
    pivot_wider(names_from = Run_Cond, values_from = c(Base, Term))
cat("Baseline Delta:  ", mean(subj_deltas_sim$`Base_2_DeepStay` - subj_deltas_sim$`Base_1_Switch`, na.rm=TRUE)*1000, "ms\n")
cat("Terminal Delta:  ", mean(subj_deltas_sim$`Term_2_DeepStay` - subj_deltas_sim$`Term_1_Switch`, na.rm=TRUE)*1000, "ms\n")
