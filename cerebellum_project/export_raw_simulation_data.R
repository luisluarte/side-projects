pacman::p_load(tidyverse, Rcpp)

Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp") 
Rcpp::sourceCpp("src/models/magi_terminal_decoupled_hybrid.cpp") 

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0),
           CumulativeFatigue = row_number()) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- 30
d_sub <- dat_clean %>% filter(participant_idx <= S)
hyper <- c(0.01, 1.00, 2.0)

cat("Fitting Baseline across N=30...\n")
obj_base <- function(phi) {
    err <- 0
    for(s in 1:S) {
        d <- d_sub %>% filter(participant_idx == s)
        sim <- extract_baseline_wald_sim(phi, d$Boundary+1, d$`F`, d$RT)
        err <- err + mean(abs(sort(sim) - sort(d$RT)))
    }
    return(err / S)
}
res_base <- optim(rep(0, 4), obj_base, method="Nelder-Mead", control=list(maxit=100))

cat("Fitting Terminal Model across N=30...\n")
obj_var <- function(phi) {
    err <- 0
    for(s in 1:S) {
        d <- d_sub %>% filter(participant_idx == s)
        sim <- extract_epoch10_2_hybrid(phi, hyper, d$Boundary+1, d$`F`, d$RT)
        err <- err + mean(abs(sort(sim) - sort(d$RT)))
    }
    return(err / S)
}
res_var <- optim(rep(0, 12), obj_var, method="Nelder-Mead", control=list(maxit=100))

cat("Generating Raw Output DataFrames...\n")
raw_rt_data <- data.frame()
raw_w1_data <- data.frame()

for(s in 1:S) {
    d <- d_sub %>% filter(participant_idx == s)
    sim_base <- extract_baseline_wald_sim(res_base$par, d$Boundary+1, d$`F`, d$RT)
    sim_var <- extract_epoch10_2_hybrid(res_var$par, hyper, d$Boundary+1, d$`F`, d$RT)
    
    # Trial-level dataframe for LMER and Plots
    tmp_rt <- data.frame(
        SubjectID = s,
        CumulativeFatigue = d$CumulativeFatigue,
        Empirical_RT = d$RT,
        Baseline_Wald_Sim_RT = sim_base,
        Terminal_Hybrid_Sim_RT = sim_var
    )
    raw_rt_data <- bind_rows(raw_rt_data, tmp_rt)
    
    # Subject-level W1 metric dataframe
    w1_base <- mean(abs(sort(sim_base) - sort(d$RT)))
    w1_var  <- mean(abs(sort(sim_var) - sort(d$RT)))
    
    tmp_w1 <- data.frame(
        SubjectID = s,
        W1_Baseline_Wald = w1_base,
        W1_Terminal_Hybrid = w1_var
    )
    raw_w1_data <- bind_rows(raw_w1_data, tmp_w1)
}

write_csv(raw_rt_data, "results/tables/magi_raw_trial_simulations_N30.csv")
write_csv(raw_w1_data, "results/tables/magi_raw_subject_W1_N30.csv")

# Export the optimized parameters to avoid re-running next time
write_csv(data.frame(Param=paste0("phi_", 1:4), Value=res_base$par), "results/tables/magi_baseline_opt_params.csv")
write_csv(data.frame(Param=paste0("phi_", 1:12), Value=res_var$par), "results/tables/magi_terminal_hybrid_opt_params.csv")

cat("Data successfully exported to results/tables/.\n")
