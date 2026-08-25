pacman::p_load(tidyverse, Rcpp, optimx, lme4, lmerTest, emmeans, parallel)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/extract_ql_fatigue.cpp")
Rcpp::sourceCpp("src/models/eccm_lc_volatility_gate.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           ITI = (ttp - lag(ttF)) / 1000, 
           F_dur = (ttF - ttr) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), 
           F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% 
    ungroup() %>% 
    filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 
num_cores <- detectCores() - 1
best_N <- 20

cat("Phase 2 & 3: Unconstrained Optimization & Dual-Gate Evaluation...\n")

final_eval <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    iti <- d$ITI
    f_dur <- d$F_dur
    ttp <- d$ttp
    T_trials <- nrow(d)
    
    # Baseline: Dynamic QL Poly Fatigue (9 params)
    obj_ql <- function(phi) { return(eval_ql_ddm_dynamic_poly_fatigue(phi, resp, out, rt) + lambda * sum(phi^2)) }
    res_ql <- optim(rep(0, 9), obj_ql, method="L-BFGS-B", lower=rep(-5, 9), upper=rep(5, 9))
    ll_ql <- extract_ll_ql_dynamic_poly_fatigue(res_ql$par, resp, out, rt)
    rt_ql <- extract_rt_ql_dynamic_poly_fatigue(res_ql$par, resp, out, rt)
    
    # Candidate: LC Volatility Gate (13 params, N=20)
    obj_lc <- function(phi) { return(eval_lc_volatility_gate(phi, resp, out, rt, iti, f_dur, ttp, best_N) + lambda * sum(phi^2)) }
    res_lc <- optim(rep(0, 13), obj_lc, method="L-BFGS-B", lower=rep(-5, 13), upper=rep(5, 13))
    ll_lc <- extract_ll_lc_volatility_gate(res_lc$par, resp, out, rt, iti, f_dur, ttp, best_N)
    rt_lc <- extract_rt_lc_volatility_gate(res_lc$par, resp, out, rt, iti, f_dur, ttp, best_N)
    
    df_ql <- data.frame(Participant_ID = d$participant_id, Trial = 1:T_trials, 
                        RT_empirical = rt, RT_predicted = rt_ql, LL = ll_ql, Model = "Dynamic_QL_Poly_Fatigue")
    df_lc <- data.frame(Participant_ID = d$participant_id, Trial = 1:T_trials, 
                        RT_empirical = rt, RT_predicted = rt_lc, LL = ll_lc, Model = "LC_Volatility_Gate")
    
    # Pointwise AIC-adjusted LL difference
    m_i <- (ll_lc - (13/T_trials)) - (ll_ql - (9/T_trials))
    
    return(list(df = bind_rows(df_ql, df_lc), m_i = m_i))
}, mc.cores = num_cores)

df_all <- bind_rows(lapply(final_eval, function(x) x$df))
m_all <- unlist(lapply(final_eval, function(x) x$m_i))

m_all <- m_all[is.finite(m_all)]
Z_stat <- sum(m_all) / (sd(m_all) * sqrt(length(m_all)))

cat("\n==========================================\n")
cat("DUAL-GATE METRICS (LC Volatility vs Baseline)\n")
cat("==========================================\n")
cat("Vuong Z-Statistic:", Z_stat, "\n")

df_filtered <- df_all %>% filter(is.finite(RT_predicted) & RT_predicted > 0 & RT_predicted < 10)
df_filtered$Model <- factor(df_filtered$Model, levels = c("Dynamic_QL_Poly_Fatigue", "LC_Volatility_Gate"))
df_filtered$Participant_ID <- as.factor(df_filtered$Participant_ID)

lmm_rt <- lmer(RT_empirical ~ Model * RT_predicted + (1 | Participant_ID), data = df_filtered)
emm <- emtrends(lmm_rt, pairwise ~ Model, var = "RT_predicted")
print(emm)

