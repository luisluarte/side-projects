pacman::p_load(tidyverse, Rcpp, cmaes)

Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp") 
Rcpp::sourceCpp("src/models/magi_terminal_decoupled_hybrid.cpp") 
Rcpp::sourceCpp("extract_expectations.cpp")

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

obj_base <- function(phi) {
    err <- 0
    for(s in 1:S) {
        d <- d_sub %>% filter(participant_idx == s)
        sim <- extract_baseline_wald_sim(phi, d$Boundary+1, d$`F`, d$RT)
        err <- err + mean(abs(sort(sim) - sort(d$RT)))
    }
    return(err / S)
}

obj_var <- function(phi) {
    err <- 0
    for(s in 1:S) {
        d <- d_sub %>% filter(participant_idx == s)
        sim <- extract_epoch10_2_hybrid(phi, hyper, d$Boundary+1, d$`F`, d$RT)
        err <- err + mean(abs(sort(sim) - sort(d$RT)))
    }
    return(err / S)
}

cat("Starting Heavy CMA-ES for Baseline Wald (N=30)...\n")
set.seed(42)
res_base <- cma_es(rep(0, 4), obj_base, control=list(maxit=300, sigma=0.5))

cat("Starting Heavy CMA-ES for Terminal Decoupled Hybrid (N=30)...\n")
set.seed(42)
res_var <- cma_es(rep(0, 12), obj_var, control=list(maxit=800, sigma=0.5))

cat("CMA-ES Complete.\nGlobal W1 Base:", res_base$value, "\nGlobal W1 Terminal:", res_var$value, "\n")

cat("Generating Raw Output DataFrames...\n")
raw_rt_data <- data.frame()
raw_w1_data <- data.frame()

p_base <- res_base$par
p_var <- res_var$par

for(s in 1:S) {
    d <- d_sub %>% filter(participant_idx == s)
    base_res <- get_base_expect(p_base, d$Boundary+1, d$`F`, d$RT)
    var_res  <- get_hybrid_expect(p_var, hyper, d$Boundary+1, d$`F`, d$RT)
    
    sim_base <- base_res[,1]
    sim_var <- var_res[,1]
    
    tmp_rt <- data.frame(
        SubjectID = s,
        CumulativeFatigue = d$CumulativeFatigue,
        Empirical_RT = d$RT,
        Baseline_Wald_Sim_RT = sim_base,
        Baseline_Wald_Expected_RT = base_res[,2],
        Terminal_Hybrid_Sim_RT = sim_var,
        Terminal_Hybrid_Expected_RT = var_res[,2]
    )
    raw_rt_data <- bind_rows(raw_rt_data, tmp_rt)
    
    w1_base <- mean(abs(sort(sim_base) - sort(d$RT)))
    w1_var  <- mean(abs(sort(sim_var) - sort(d$RT)))
    
    tmp_w1 <- data.frame(
        SubjectID = s,
        W1_Baseline_Wald = w1_base,
        W1_Terminal_Hybrid = w1_var
    )
    raw_w1_data <- bind_rows(raw_w1_data, tmp_w1)
}

write_csv(raw_rt_data, "results/tables/magi_raw_trial_simulations_N30_CMAES.csv")
write_csv(raw_w1_data, "results/tables/magi_raw_subject_W1_N30_CMAES.csv")

write_csv(data.frame(Param=paste0("phi_", 1:4), Value=p_base), "results/tables/magi_baseline_opt_params_CMAES.csv")
write_csv(data.frame(Param=paste0("phi_", 1:12), Value=p_var), "results/tables/magi_terminal_hybrid_opt_params_CMAES.csv")

cat("SUCCESS: All heavy CMA-ES exports saved to results/tables/ (appended with _CMAES).\n")
