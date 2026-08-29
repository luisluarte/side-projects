pacman::p_load(tidyverse, doParallel, foreach, zoo, TTR)

sim_data <- read_csv("results/tables/magi_raw_trial_simulations_N30_CMAES.csv", show_col_types = FALSE)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
p_base <- read_csv("results/tables/magi_baseline_opt_params_CMAES.csv", show_col_types=FALSE)$Value
p_var <- read_csv("results/tables/magi_terminal_hybrid_opt_params_CMAES.csv", show_col_types=FALSE)$Value
alpha_base <- 1.0 / (1.0 + exp(-p_base[4]))
alpha_term <- 1.0 / (1.0 + exp(-p_var[4]))

dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0), Reward = `F`) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id))) %>% filter(participant_idx <= 30)

calc_probs <- function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    n <- nrow(d)
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
    data.frame(SubjectID=s_idx, Trial=1:n, Is_Switch=is_sw, P_Switch_Base=p_b, P_Switch_Term=p_t)
}

prob_df <- bind_rows(lapply(1:30, calc_probs))
sim_data <- sim_data %>% group_by(SubjectID) %>% mutate(Trial = row_number()) %>% ungroup() %>%
    left_join(prob_df, by=c("SubjectID", "Trial"))
    
calc_objectives <- function(df) {
    w1_base <- mean(abs(sort(df$Baseline_Wald_Sim_RT) - sort(df$Empirical_RT)))
    w1_term <- mean(abs(sort(df$Terminal_Hybrid_Sim_RT) - sort(df$Empirical_RT)))
    
    get_slope <- function(x, y) { cov(x, y)/var(x) }
    emp_slope <- get_slope(df$CumulativeFatigue, df$Empirical_RT)
    base_slope <- get_slope(df$CumulativeFatigue, df$Baseline_Wald_Sim_RT)
    term_slope <- get_slope(df$CumulativeFatigue, df$Terminal_Hybrid_Sim_RT)
    db_base <- abs(emp_slope - base_slope)
    db_term <- abs(emp_slope - term_slope)
    
    df_sw <- df %>% filter(!is.na(Is_Switch))
    eps <- 1e-15
    p_b <- pmax(pmin(df_sw$P_Switch_Base, 1-eps), eps)
    p_t <- pmax(pmin(df_sw$P_Switch_Term, 1-eps), eps)
    ll_base <- -mean(df_sw$Is_Switch * log(p_b) + (1-df_sw$Is_Switch) * log(1-p_b))
    ll_term <- -mean(df_sw$Is_Switch * log(p_t) + (1-df_sw$Is_Switch) * log(1-p_t))
    
    df_ema <- df %>% group_by(SubjectID) %>% mutate(
        ema_emp = EMA(Empirical_RT, n=10),
        ema_base = EMA(Baseline_Wald_Expected_RT, n=10),
        ema_term = EMA(Terminal_Hybrid_Expected_RT, n=10)
    ) %>% ungroup() %>% filter(!is.na(ema_emp))
    phase_base <- mean((df_ema$ema_emp - df_ema$ema_base)^2)
    phase_term <- mean((df_ema$ema_emp - df_ema$ema_term)^2)
    
    return(c(w1_base, db_base, ll_base, phase_base, w1_term, db_term, ll_term, phase_term))
}

cat("Initializing 12-core cluster for 1000 Bootstraps...\n")
cl <- makeCluster(12)
registerDoParallel(cl)

set.seed(42)
boot_res <- foreach(b = 1:1000, .combine=rbind, .packages=c("tidyverse", "TTR")) %dopar% {
    sampled_subs <- sample(1:30, 30, replace=TRUE)
    boot_df <- bind_rows(lapply(1:30, function(i) {
        d <- sim_data %>% filter(SubjectID == sampled_subs[i])
        d$SubjectID <- i 
        d
    }))
    calc_objectives(boot_df)
}
stopCluster(cl)

boot_df <- as.data.frame(boot_res)
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

p_val <- mean(boot_df$Delta_HV <= 0)

eps_indicator <- apply(boot_df, 1, function(row) {
    max(c(row["W1_B"] - row["W1_T"], 
          row["dB_B"] - row["dB_T"], 
          row["LL_B"] - row["LL_T"], 
          row["Ph_B"] - row["Ph_T"]))
})
mean_eps <- mean(eps_indicator)

cat("=== BOOTSTRAPPED PARETO FRONTIER STATISTICS (B=1000) ===\n")
cat("Reference Point Z_ref:", Z_ref, "\n")
cat(sprintf("Mean HV Baseline: %g\n", mean(boot_df$HV_Base)))
cat(sprintf("Mean HV Terminal Hybrid: %g\n", mean(boot_df$HV_Term)))
cat(sprintf("Mean Delta HV: %g\n", mean(boot_df$Delta_HV)))
cat(sprintf("Bootstrapped P-Value of Supremacy: %.4f\n", p_val))
cat(sprintf("Additive Epsilon-Indicator (Base -> Term): %.6f\n", mean_eps))

write_csv(boot_df, "results/tables/magi_pareto_bootstrap_B1000.csv")
cat("SUCCESS: Bootstrapped Pareto sets saved.\n")
