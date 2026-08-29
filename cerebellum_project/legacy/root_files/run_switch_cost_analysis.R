pacman::p_load(tidyverse, lme4, lmerTest)

sim_data <- read_csv("results/tables/magi_raw_trial_simulations_N30.csv", show_col_types = FALSE)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)

dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% 
    mutate(participant_idx = as.integer(as.factor(participant_id))) %>% 
    filter(participant_idx <= 30) %>%
    group_by(participant_idx) %>%
    mutate(Is_Switch = ifelse(row_number() > 1 & Resp != lag(Resp), 1, 0)) %>%
    ungroup()

# Align data
sim_data$Is_Switch <- dat_clean$Is_Switch
sim_data <- sim_data %>% filter(!is.na(Is_Switch))
sim_data$Is_Switch <- factor(sim_data$Is_Switch, levels=c(0, 1), labels=c("Stay", "Switch"))

# Calculate Subject Level Deltas (Switch Cost)
subj_deltas <- sim_data %>%
    group_by(SubjectID, Is_Switch) %>%
    summarize(
        Emp = mean(Empirical_RT),
        Base = mean(Baseline_Wald_Sim_RT),
        Term = mean(Terminal_Hybrid_Sim_RT),
        .groups = "drop"
    ) %>%
    pivot_wider(names_from = Is_Switch, values_from = c(Emp, Base, Term)) %>%
    mutate(
        Delta_Emp = Emp_Switch - Emp_Stay,
        Delta_Base = Base_Switch - Base_Stay,
        Delta_Term = Term_Switch - Term_Stay
    )

cat("=== MEAN SWITCH COSTS (DELTA RT) ===\n")
cat("Empirical Switch Cost: ", mean(subj_deltas$Delta_Emp) * 1000, " ms\n")
cat("Baseline Model Switch Cost: ", mean(subj_deltas$Delta_Base) * 1000, " ms\n")
cat("Terminal Hybrid Switch Cost: ", mean(subj_deltas$Delta_Term) * 1000, " ms\n\n")

cat("=== PAIRED T-TEST: EMPIRICAL vs BASELINE DELTA ===\n")
print(t.test(subj_deltas$Delta_Emp, subj_deltas$Delta_Base, paired=TRUE))

cat("=== PAIRED T-TEST: EMPIRICAL vs TERMINAL HYBRID DELTA ===\n")
print(t.test(subj_deltas$Delta_Emp, subj_deltas$Delta_Term, paired=TRUE))
