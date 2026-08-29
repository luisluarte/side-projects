pacman::p_load(tidyverse, Rcpp, cmaes, parallel, TTR)

CORES <- parallel::detectCores()
hyper <- c(0.01, 1.00, 2.0)

Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp") 
Rcpp::sourceCpp("src/models/magi_terminal_decoupled_hybrid.cpp") 
Rcpp::sourceCpp("extract_expectations.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% dplyr::filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_sub <- dat_clean %>% dplyr::filter(participant_idx <= S)
d_list <- split(d_sub, d_sub$participant_idx)

hybrid_mat <- read_csv("results/tables/magi_subject_level_hybrid_S128_matrix.csv", show_col_types=FALSE)

cat("Fitting Individual Baseline Walds...\n")
run_base_opt <- function(s_idx) {
    d <- d_list[[s_idx]]
    obj_base <- function(phi) {
        sim <- extract_baseline_wald_sim(phi, d$Boundary+1, d$Reward, d$RT)
        w1 <- mean(abs(sort(sim) - sort(d$RT)))
        
        # We should use the same joint objective for fairness: W1 + Phase
        var_res <- get_base_expect(phi, d$Boundary+1, d$Reward, d$RT)
        exp_t <- var_res[,2]
        
        if(nrow(d) > 10) {
            ema_emp <- EMA(d$RT, n=10)
            ema_term <- EMA(exp_t, n=10)
            valid <- !is.na(ema_emp) & !is.na(ema_term)
            if(sum(valid) > 0) {
                phase <- mean((ema_emp[valid] - ema_term[valid])^2)
            } else { phase <- 1.0 }
        } else { phase <- 1.0 }
        
        return(w1 + phase)
    }
    res <- cma_es(rep(0, 4), obj_base, control=list(maxit=300, sigma=0.5))
    return(c(SubjectID = s_idx, res$par, Cost = res$value))
}

base_res_list <- mclapply(1:S, run_base_opt, mc.cores = CORES)
base_mat <- as.data.frame(do.call(rbind, base_res_list))

cat("Generating Individualized Data Matrices...\n")
gen_data <- function(s_idx) {
    d <- d_list[[s_idx]]
    p_b <- as.numeric(base_mat[s_idx, 2:5])
    p_t <- as.numeric(hybrid_mat[s_idx, 2:13])
    
    sim_b <- extract_baseline_wald_sim(p_b, d$Boundary+1, d$Reward, d$RT)
    exp_b_res <- get_base_expect(p_b, d$Boundary+1, d$Reward, d$RT)
    exp_b <- exp_b_res[,2]
    
    sim_t <- extract_epoch10_2_hybrid(p_t, hyper, d$Boundary+1, d$Reward, d$RT)
    exp_t_res <- get_hybrid_expect(p_t, hyper, d$Boundary+1, d$Reward, d$RT)
    exp_t <- exp_t_res[,2]
    
    # Probabilities for logloss
    is_sw <- rep(NA, nrow(d))
    p_b_sw <- rep(0.5, nrow(d))
    p_t_sw <- rep(0.5, nrow(d))
    for(t in 2:nrow(d)) {
        is_sw[t] <- ifelse(d$Boundary[t] != d$Boundary[t-1], 1, 0)
        p_b_sw[t] <- 1 - exp_b_res[t, d$Boundary[t-1]+1]
        p_t_sw[t] <- 1 - exp_t_res[t, d$Boundary[t-1]+1]
    }
    
    return(data.frame(
        SubjectID=s_idx, Trial=1:nrow(d), Empirical_RT=d$RT,
        Baseline_Wald_Sim_RT=sim_b, Baseline_Wald_Expected_RT=exp_b,
        Terminal_Hybrid_Sim_RT=sim_t, Terminal_Hybrid_Expected_RT=exp_t,
        Is_Switch=is_sw, P_Switch_Base=p_b_sw, P_Switch_Term=p_t_sw
    ))
}
sim_results <- mclapply(1:S, gen_data, mc.cores = CORES)
df_sim <- bind_rows(sim_results)

get_spectral_beta <- function(x) {
    s <- spectrum(x, plot=FALSE)
    coef(lm(log(s$spec) ~ log(s$freq)))[2]
}

calc_objectives <- function(df) {
    w1_base <- mean(abs(sort(df$Baseline_Wald_Sim_RT) - sort(df$Empirical_RT)))
    w1_term <- mean(abs(sort(df$Terminal_Hybrid_Sim_RT) - sort(df$Empirical_RT)))
    
    emp_beta <- get_spectral_beta(df$Empirical_RT)
    base_beta <- get_spectral_beta(df$Baseline_Wald_Expected_RT)
    term_beta <- get_spectral_beta(df$Terminal_Hybrid_Expected_RT)
    db_base <- abs(emp_beta - base_beta)
    db_term <- abs(emp_beta - term_beta)
    
    df_sw <- df %>% dplyr::filter(!is.na(Is_Switch))
    eps <- 1e-15
    p_b_v <- pmax(pmin(df_sw$P_Switch_Base, 1-eps), eps)
    p_t_v <- pmax(pmin(df_sw$P_Switch_Term, 1-eps), eps)
    ll_base <- -mean(df_sw$Is_Switch * log(p_b_v) + (1-df_sw$Is_Switch) * log(1-p_b_v))
    ll_term <- -mean(df_sw$Is_Switch * log(p_t_v) + (1-df_sw$Is_Switch) * log(1-p_t_v))
    
    df_ema <- df %>% group_by(SubjectID) %>% mutate(
        ema_emp = EMA(Empirical_RT, n=10),
        ema_base = EMA(Baseline_Wald_Expected_RT, n=10),
        ema_term = EMA(Terminal_Hybrid_Expected_RT, n=10)
    ) %>% ungroup() %>% dplyr::filter(!is.na(ema_emp))
    phase_base <- mean((df_ema$ema_emp - df_ema$ema_base)^2)
    phase_term <- mean((df_ema$ema_emp - df_ema$ema_term)^2)
    
    return(c(w1_base, db_base, ll_base, phase_base, w1_term, db_term, ll_term, phase_term))
}

cat("Running Pareto Spectral Bootstrap (Subject-Level)...\n")
boot_res_list <- mclapply(1:1000, function(b) {
    sampled_subs <- sample(1:S, S, replace=TRUE)
    boot_df <- bind_rows(lapply(1:S, function(i) {
        df_sim %>% filter(SubjectID == sampled_subs[i]) %>% mutate(SubjectID = i)
    }))
    calc_objectives(boot_df)
}, mc.cores = CORES)

boot_df <- as.data.frame(do.call(rbind, boot_res_list))
colnames(boot_df) <- c("W1_B", "dB_B", "LL_B", "Ph_B", "W1_T", "dB_T", "LL_T", "Ph_T")

Z_ref <- c(
    W1 = max(c(boot_df$W1_B, boot_df$W1_T)) * 1.1,
    dB = max(c(boot_df$dB_B, boot_df$dB_T)) * 1.1,
    LL = max(c(boot_df$LL_B, boot_df$LL_T)) * 1.1,
    Ph = max(c(boot_df$Ph_B, boot_df$Ph_T)) * 1.1
)

boot_df <- boot_df %>% mutate(
    HV_Base = (Z_ref["W1"] - W1_B) * (Z_ref["dB"] - dB_B) * (Z_ref["LL"] - LL_B) * (Z_ref["Ph"] - Ph_B),
    HV_Term = (Z_ref["W1"] - W1_T) * (Z_ref["dB"] - dB_T) * (Z_ref["LL"] - LL_T) * (Z_ref["Ph"] - Ph_T),
    Delta_HV = HV_Term - HV_Base
)

global_w1_base <- mean(abs(sort(df_sim$Empirical_RT) - sort(df_sim$Baseline_Wald_Sim_RT)))
global_w1_term <- mean(abs(sort(df_sim$Empirical_RT) - sort(df_sim$Terminal_Hybrid_Sim_RT)))

sink("results/tables/magi_final_N128_subject_stats.txt")
cat("=== N=128 SUBJECT-LEVEL GCP SUPERCOMPUTER VALIDATION ===\n")
cat("Global W1 Base:", global_w1_base, "\n")
cat("Global W1 Terminal Hybrid:", global_w1_term, "\n")
cat("P-Value of Supremacy (Base < Hybrid):", mean(boot_df$Delta_HV <= 0), "\n")
cat("Mean Delta HV:", mean(boot_df$Delta_HV), "\n")
cat("Epsilon Indicator:", mean(apply(boot_df, 1, function(row) max(c(row[1]-row[5], row[2]-row[6], row[3]-row[7], row[4]-row[8])))), "\n")
sink()
cat("--- ALL DONE! ---\n")
