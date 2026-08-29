pacman::p_load(tidyverse, Rcpp)
Rcpp::sourceCpp("extract_expectations.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0),
           CumulativeFatigue = row_number()) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id))) %>% filter(participant_idx <= 30)

p_base <- read_csv("results/tables/magi_baseline_opt_params.csv", show_col_types = FALSE)$Value
p_var <- read_csv("results/tables/magi_terminal_hybrid_opt_params.csv", show_col_types = FALSE)$Value

hyper <- c(0.01, 1.00, 2.0)
raw_rt_data <- data.frame()

for(s in 1:30) {
    d <- dat_clean %>% filter(participant_idx == s)
    base_res <- get_base_expect(p_base, d$Boundary+1, d$`F`, d$RT)
    var_res  <- get_hybrid_expect(p_var, hyper, d$Boundary+1, d$`F`, d$RT)
    
    tmp_rt <- data.frame(
        SubjectID = s,
        CumulativeFatigue = d$CumulativeFatigue,
        Empirical_RT = d$RT,
        Baseline_Wald_Sim_RT = base_res[,1],
        Baseline_Wald_Expected_RT = base_res[,2],
        Terminal_Hybrid_Sim_RT = var_res[,1],
        Terminal_Hybrid_Expected_RT = var_res[,2]
    )
    raw_rt_data <- bind_rows(raw_rt_data, tmp_rt)
}

write_csv(raw_rt_data, "results/tables/magi_raw_trial_simulations_N30.csv")
cat("SUCCESS: Added deterministic expectations to CSV.\n")
