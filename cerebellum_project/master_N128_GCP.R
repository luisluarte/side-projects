pacman::p_load(tidyverse, Rcpp, cmaes, parallel, pROC, PRROC, pracma, TTR, signal, doParallel, foreach)

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

S <- 128
d_sub <- dat_clean %>% dplyr::filter(participant_idx <= S)
hyper <- c(0.01, 1.00, 2.0)
CORES <- 32

cat("--- PHASE 1: N=128 PARALLEL CMA-ES ON 32 CORES ---\n")
d_list <- split(d_sub, d_sub$participant_idx)

obj_base <- function(phi) {
    errs <- mclapply(1:S, function(s) {
        d <- d_list[[s]]
        sim <- extract_baseline_wald_sim(phi, d$Boundary+1, d$Reward, d$RT)
        mean(abs(sort(sim) - sort(d$RT)))
    }, mc.cores = CORES)
    return(mean(unlist(errs)))
}

obj_var <- function(phi) {
    errs <- mclapply(1:S, function(s) {
        d <- d_list[[s]]
        sim <- extract_epoch10_2_hybrid(phi, hyper, d$Boundary+1, d$Reward, d$RT)
        mean(abs(sort(sim) - sort(d$RT)))
    }, mc.cores = CORES)
    return(mean(unlist(errs)))
}

set.seed(42)
cat("Optimizing Baseline Wald (N=128, 300 iters)...\n")
res_base <- cma_es(rep(0, 4), obj_base, control=list(maxit=300, sigma=0.5))

cat("Optimizing Terminal Hybrid (N=128, 800 iters)...\n")
res_var <- cma_es(rep(0, 12), obj_var, control=list(maxit=800, sigma=0.5))

p_base <- res_base$par
p_var <- res_var$par

cat("Global W1 Base:", res_base$value, "\nGlobal W1 Terminal:", res_var$value, "\n")

cat("--- PHASE 2: GENERATING DATA MATRICES ---\n")
alpha_base <- 1.0 / (1.0 + exp(-p_base[4]))
alpha_term <- 1.0 / (1.0 + exp(-p_var[4]))

sim_results <- mclapply(1:S, function(s) {
    d <- d_list[[s]]
    n <- nrow(d)
    
    base_res <- get_base_expect(p_base, d$Boundary+1, d$Reward, d$RT)
    var_res  <- get_hybrid_expect(p_var, hyper, d$Boundary+1, d$Reward, d$RT)
    
    sim_b <- base_res[,1]; exp_b <- base_res[,2]
    sim_t <- var_res[,1];  exp_t <- var_res[,2]
    
    w1_b <- mean(abs(sort(sim_b) - sort(d$RT)))
    w1_t <- mean(abs(sort(sim_t) - sort(d$RT)))
    
    p_b <- numeric(n); p_t <- numeric(n)
    is_sw <- numeric(n)
    Qb <- c(0.5, 0.5); Qt <- c(0.5, 0.5)
    for(t in 1:n) {
        ch <- d$Boundary[t] + 1
        r <- d$Reward[t]
        if(t > 1) {
            prev_ch <- d$Boundary[t-1] + 1
            is_sw[t] <- ifelse(ch != prev_ch, 1, 0)
            p_b[t] <- 1 - (Qb[prev_ch] / sum(Qb))
            p_t[t] <- 1 - (Qt[prev_ch] / sum(Qt))
        } else {
            is_sw[t] <- NA; p_b[t] <- NA; p_t[t] <- NA
        }
        Qb[ch] <- Qb[ch] + alpha_base * (r - Qb[ch])
        Qt[ch] <- Qt[ch] + alpha_term * (r - Qt[ch])
    }
    
    list(
        rt_df = data.frame(SubjectID=s, Trial=1:n, 
                           Empirical_RT=d$RT, Baseline_Wald_Sim_RT=sim_b, Baseline_Wald_Expected_RT=exp_b,
                           Terminal_Hybrid_Sim_RT=sim_t, Terminal_Hybrid_Expected_RT=exp_t,
                           Is_Switch=is_sw, P_Switch_Base=p_b, P_Switch_Term=p_t),
        w1_df = data.frame(SubjectID=s, W1_Baseline=w1_b, W1_Terminal=w1_t)
    )
}, mc.cores = CORES)

raw_rt_data <- bind_rows(lapply(sim_results, function(x) x$rt_df))
raw_w1_data <- bind_rows(lapply(sim_results, function(x) x$w1_df))

write_csv(raw_rt_data, "results/tables/magi_raw_trial_simulations_N128_GCP.csv")
write_csv(raw_w1_data, "results/tables/magi_raw_subject_W1_N128_GCP.csv")
write_csv(data.frame(Param=paste0("phi_", 1:4), Value=p_base), "results/tables/magi_baseline_opt_params_N128_GCP.csv")
write_csv(data.frame(Param=paste0("phi_", 1:12), Value=p_var), "results/tables/magi_terminal_hybrid_opt_params_N128_GCP.csv")

cat("--- PHASE 3: SPECTRAL PARETO BOOTSTRAP (B=1000) ---\n")

get_spectral_beta <- function(x) {
    if(sd(x) < 1e-6) return(0)
    pw <- spectrum(x, plot=FALSE)
    valid <- pw$freq > 0 & pw$spec > 0
    if(sum(valid) < 2) return(0)
    fit <- lm(log(pw$spec[valid]) ~ log(pw$freq[valid]))
    return(as.numeric(-coef(fit)[2]))
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
    p_b <- pmax(pmin(df_sw$P_Switch_Base, 1-eps), eps)
    p_t <- pmax(pmin(df_sw$P_Switch_Term, 1-eps), eps)
    ll_base <- -mean(df_sw$Is_Switch * log(p_b) + (1-df_sw$Is_Switch) * log(1-p_b))
    ll_term <- -mean(df_sw$Is_Switch * log(p_t) + (1-df_sw$Is_Switch) * log(1-p_t))
    
    df_ema <- df %>% group_by(SubjectID) %>% mutate(
        ema_emp = EMA(Empirical_RT, n=10),
        ema_base = EMA(Baseline_Wald_Expected_RT, n=10),
        ema_term = EMA(Terminal_Hybrid_Expected_RT, n=10)
    ) %>% ungroup() %>% dplyr::filter(!is.na(ema_emp))
    phase_base <- mean((df_ema$ema_emp - df_ema$ema_base)^2)
    phase_term <- mean((df_ema$ema_emp - df_ema$ema_term)^2)
    
    return(c(w1_base, db_base, ll_base, phase_base, w1_term, db_term, ll_term, phase_term))
}

boot_res_list <- mclapply(1:1000, function(b) {
    sampled_subs <- sample(1:S, S, replace=TRUE)
    boot_df <- bind_rows(lapply(1:S, function(i) {
        d <- raw_rt_data[raw_rt_data$SubjectID == sampled_subs[i], ]
        d$SubjectID <- i 
        d
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

write_csv(boot_df, "results/tables/magi_pareto_spectral_bootstrap_N128_GCP.csv")

sink("results/tables/magi_final_N128_stats.txt")
cat("=== N=128 GCP SUPERCOMPUTER VALIDATION ===\n")
cat("P-Value of Supremacy:", mean(boot_df$Delta_HV <= 0), "\n")
cat("Mean Delta HV:", mean(boot_df$Delta_HV), "\n")
cat("Epsilon Indicator:", mean(apply(boot_df, 1, function(row) max(c(row[1]-row[5], row[2]-row[6], row[3]-row[7], row[4]-row[8])))), "\n")
sink()
cat("--- ALL DONE! ---\n")
