pacman::p_load(tidyverse, lme4, lmerTest)

sim_data <- read_csv("results/tables/magi_raw_trial_simulations_N30.csv", show_col_types = FALSE)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)

dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0),
           Reward = `F`) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward))

dat_clean <- dat_clean %>% 
    mutate(participant_idx = as.integer(as.factor(participant_id))) %>% 
    filter(participant_idx <= 30) %>%
    group_by(participant_idx) %>%
    mutate(
        Prev_Reward = lag(Reward),
        # Fatigue Quantiles
        Fatigue_Q = ntile(row_number(), 5),
        Condition_Fatigue = case_when(Fatigue_Q == 1 ~ "1_Early", Fatigue_Q == 5 ~ "2_Late", TRUE ~ NA_character_),
        # Post Error Slowing
        Condition_PES = case_when(Prev_Reward == 1 ~ "1_PostWin", Prev_Reward == 0 ~ "2_PostLoss", TRUE ~ NA_character_),
        # Cumulative Streak
        Streak = sequence(rle(Reward)$lengths) * ifelse(Reward==1, 1, -1),
        Prev_Streak = lag(Streak),
        Condition_Streak = case_when(Prev_Streak >= 3 ~ "1_Post3Wins", Prev_Streak <= -3 ~ "2_Post3Losses", TRUE ~ NA_character_)
    ) %>%
    ungroup()

sim_data$Condition_Fatigue <- dat_clean$Condition_Fatigue
sim_data$Condition_PES <- dat_clean$Condition_PES
sim_data$Condition_Streak <- dat_clean$Condition_Streak

analyze_condition <- function(data, cond_col, cond_name) {
    cat(paste0("\n========================================\n"))
    cat(paste0("CONDITION: ", cond_name, "\n"))
    
    df <- data %>% filter(!is.na(.data[[cond_col]]))
    
    means <- df %>% group_by(.data[[cond_col]]) %>% 
        summarize(Emp = mean(Empirical_RT), Base = mean(Baseline_Wald_Sim_RT), Term = mean(Terminal_Hybrid_Sim_RT))
    print(means)
    
    f <- as.formula(paste0("Empirical_RT ~ ", cond_col, " + (1|SubjectID)"))
    mod <- lmer(f, data=df)
    cat("\n--- Empirical LMER Significance ---\n")
    print(summary(mod)$coefficients)
    
    lvls <- sort(unique(df[[cond_col]]))
    if(length(lvls) == 2) {
        subj_deltas <- df %>% group_by(SubjectID, .data[[cond_col]]) %>%
            summarize(Emp=mean(Empirical_RT), Base=mean(Baseline_Wald_Sim_RT), Term=mean(Terminal_Hybrid_Sim_RT), .groups="drop") %>%
            pivot_wider(names_from = .data[[cond_col]], values_from = c(Emp, Base, Term))
        
        col_emp1 <- paste0("Emp_", lvls[1]); col_emp2 <- paste0("Emp_", lvls[2])
        col_base1 <- paste0("Base_", lvls[1]); col_base2 <- paste0("Base_", lvls[2])
        col_term1 <- paste0("Term_", lvls[1]); col_term2 <- paste0("Term_", lvls[2])
        
        delta_emp <- subj_deltas[[col_emp2]] - subj_deltas[[col_emp1]]
        delta_base <- subj_deltas[[col_base2]] - subj_deltas[[col_base1]]
        delta_term <- subj_deltas[[col_term2]] - subj_deltas[[col_term1]]
        
        cat("\n--- Delta RT (", lvls[2], " - ", lvls[1], ") ---\n", sep="")
        cat("Empirical Delta: ", mean(delta_emp, na.rm=TRUE)*1000, "ms\n")
        cat("Baseline Delta:  ", mean(delta_base, na.rm=TRUE)*1000, "ms\n")
        cat("Terminal Delta:  ", mean(delta_term, na.rm=TRUE)*1000, "ms\n")
    }
}

analyze_condition(sim_data, "Condition_PES", "Post-Error Slowing (Post-Loss vs Post-Win)")
analyze_condition(sim_data, "Condition_Fatigue", "Deep Fatigue (Last 20% vs First 20% of Session)")
analyze_condition(sim_data, "Condition_Streak", "Streak Interruption (Post 3 Losses vs Post 3 Wins)")
