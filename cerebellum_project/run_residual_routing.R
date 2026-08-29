pacman::p_load(tidyverse, Rcpp, optimx, lme4, lmerTest, emmeans, parallel)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/extract_ql_fatigue.cpp")
Rcpp::sourceCpp("src/models/eccm_residual_routing.cpp")

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

N_grid <- c(5, 10, 20, 40, 80)
cat("Phase 2: Dimensionality Grid Search for Residual Routing...\n")

grid_res <- list()
for (N_val in N_grid) {
    n_res <- mclapply(1:S, function(s_idx) {
        d <- dat_clean %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1
        out <- d$`F`
        rt <- d$RT
        iti <- d$ITI
        f_dur <- d$F_dur
        ttp <- d$ttp
        
        obj_rr <- function(phi) { return(eval_residual_routing(phi, resp, out, rt, iti, f_dur, ttp, N_val) + lambda * sum(phi^2)) }
        res <- optim(rep(0, 10), obj_rr, method="L-BFGS-B", lower=rep(-5, 10), upper=rep(5, 10))
        return(data.frame(Participant = s_idx, Deviance = res$value, par = I(list(res$par))))
    }, mc.cores = num_cores)
    
    df_n <- bind_rows(n_res)
    grid_res[[length(grid_res) + 1]] <- data.frame(N_MF = N_val, Global_Deviance = sum(df_n$Deviance))
}

grid_df <- bind_rows(grid_res)
print(grid_df)
best_N <- grid_df$N_MF[which.min(grid_df$Global_Deviance)]
cat("\nOptimal N_MF discovered:", best_N, "\n")

cat("Phase 3: Extraction & Global Evaluation against Baseline...\n")

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
    
    # Residual Routing (10 params, Best N)
    obj_rr <- function(phi) { return(eval_residual_routing(phi, resp, out, rt, iti, f_dur, ttp, best_N) + lambda * sum(phi^2)) }
    res_rr <- optim(rep(0, 10), obj_rr, method="L-BFGS-B", lower=rep(-5, 10), upper=rep(5, 10))
    ll_rr <- extract_ll_residual_routing(res_rr$par, resp, out, rt, iti, f_dur, ttp, best_N)
    rt_rr <- extract_rt_residual_routing(res_rr$par, resp, out, rt, iti, f_dur, ttp, best_N)
    
    df_ql <- data.frame(Participant_ID = d$participant_id, Trial = 1:T_trials, 
                        RT_empirical = rt, RT_predicted = rt_ql, LL = ll_ql, Model = "Dynamic_QL_Poly_Fatigue")
    df_rr <- data.frame(Participant_ID = d$participant_id, Trial = 1:T_trials, 
                        RT_empirical = rt, RT_predicted = rt_rr, LL = ll_rr, Model = "Residual_Routing")
    
    # Pointwise AIC-adjusted LL difference
    m_i <- (ll_rr - (10/T_trials)) - (ll_ql - (9/T_trials))
    
    return(list(df = bind_rows(df_ql, df_rr), m_i = m_i))
}, mc.cores = num_cores)

df_all <- bind_rows(lapply(final_eval, function(x) x$df))
m_all <- unlist(lapply(final_eval, function(x) x$m_i))

m_all <- m_all[is.finite(m_all)]
Z_stat <- sum(m_all) / (sd(m_all) * sqrt(length(m_all)))

cat("\nVuong Z-Statistic (Residual Routing vs Dynamic QL Poly Fatigue):", Z_stat, "\n")

df_filtered <- df_all %>% filter(is.finite(RT_predicted) & RT_predicted > 0 & RT_predicted < 10)
df_filtered$Model <- as.factor(df_filtered$Model)
df_filtered$Participant_ID <- as.factor(df_filtered$Participant_ID)

cat("\nFitting LMM for Magnitude RT Prediction...\n")
lmm_rt <- lmer(RT_empirical ~ Model * RT_predicted + (1 | Participant_ID), data = df_filtered)
cat("\nSummary of emtrends:\n")
emm <- emtrends(lmm_rt, pairwise ~ Model, var = "RT_predicted")
print(emm)

write_csv(grid_df, "residual_routing_grid_results.csv")
