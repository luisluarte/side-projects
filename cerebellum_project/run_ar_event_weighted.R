pacman::p_load(tidyverse, Rcpp, optimx, lme4, lmerTest, parallel)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_ar_fatigue.cpp")

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
best_N <- 20
num_cores <- detectCores() - 1

cat("Phase 1: Event Indexing...\n")
dat_clean <- dat_clean %>%
    group_by(participant_id) %>%
    mutate(Trial_Idx = row_number(),
           Is_Reversal = ifelse(Trial_Idx > 1 & buena != lag(buena), TRUE, FALSE)) %>%
    ungroup()

sigma_pre_grid <- c(1, 2, 4)
sigma_post_grid <- c(4, 8, 12, 16)
grid_params <- expand.grid(Sigma_Pre = sigma_pre_grid, Sigma_Post = sigma_post_grid)

cat("Phase 2-4: AR Event-Weighted Grid Search on", num_cores, "cores...\n")

final_results <- data.frame()

for (row in 1:nrow(grid_params)) {
    s_pre <- grid_params$Sigma_Pre[row]
    s_post <- grid_params$Sigma_Post[row]
    
    cat(sprintf("Evaluating Sigma_Pre = %d, Sigma_Post = %d ...\n", s_pre, s_post))
    
    iter_res <- mclapply(1:S, function(s_idx) {
        d <- dat_clean %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1
        out <- d$`F`
        rt <- d$RT
        iti <- d$ITI
        f_dur <- d$F_dur
        ttp <- d$ttp
        T_trials <- nrow(d)
        
        R_idx <- which(d$Is_Reversal)
        
        W <- rep(0.1, T_trials)
        if (length(R_idx) > 0) {
            for (r in R_idx) {
                t_seq <- 1:T_trials
                W <- W + ifelse(t_seq < r, 
                                exp(-((t_seq - r)^2) / (2 * s_pre^2)), 
                                exp(-((t_seq - r)^2) / (2 * s_post^2)))
            }
        }
        W <- pmin(W, 1.0)
        
        # Base QL Poly (8 params)
        obj_ql <- function(phi) { 
            ll_vec <- extract_ll_ql_dynamic_poly_point(phi, resp, out, rt)
            ll_vec[!is.finite(ll_vec)] <- -1e9
            return(-sum(W * ll_vec) + lambda * sum(phi^2))
        }
        res_ql <- optim(rep(0, 8), obj_ql, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
        ll_ql <- extract_ll_ql_dynamic_poly_point(res_ql$par, resp, out, rt)
        
        # Dual FG AR Fatigue (13 params)
        obj_ar <- function(phi) { 
            ll_vec <- extract_ll_bvk_full_gating_ar_fatigue(phi, resp, out, rt, iti, f_dur, ttp, best_N)
            ll_vec[!is.finite(ll_vec)] <- -1e9
            return(-sum(W * ll_vec) + lambda * sum(phi^2))
        }
        res_ar <- optim(rep(0, 13), obj_ar, method="L-BFGS-B", lower=rep(-5, 13), upper=rep(5, 13))
        ll_ar <- extract_ll_bvk_full_gating_ar_fatigue(res_ar$par, resp, out, rt, iti, f_dur, ttp, best_N)
        pred_rt_ar <- extract_rt_bvk_full_gating_ar_fatigue(res_ar$par, resp, out, rt, iti, f_dur, ttp, best_N)
        
        df <- data.frame(
            Participant_ID = d$participant_id,
            Trial = 1:T_trials,
            Weight = W,
            RT_empirical = rt,
            RT_pred_fg = pred_rt_ar,
            LL_ql = ll_ql,
            LL_fg = ll_ar
        )
        return(df)
    }, mc.cores = num_cores)
    
    df_all <- bind_rows(iter_res)
    df_all <- df_all %>% filter(is.finite(LL_ql) & is.finite(LL_fg) & is.finite(RT_pred_fg))
    df_all <- df_all %>% mutate(Delta_LL = LL_fg - LL_ql)
    
    df_high_w <- df_all %>% filter(Weight > 0.5 & RT_pred_fg > 0 & RT_pred_fg < 10)
    
    beta_rt_val <- NA
    if (nrow(df_high_w) > 50) {
        lmm_rt <- lmer(RT_empirical ~ RT_pred_fg + (1 | Participant_ID), data = df_high_w)
        beta_rt_val <- coef(summary(lmm_rt))["RT_pred_fg", "Estimate"]
    }
    
    beta_ll_val <- NA
    t_stat_ll <- NA
    if (nrow(df_all) > 50) {
        lmm_ll <- lmer(Delta_LL ~ 1 + (1 | Participant_ID), data = df_all)
        beta_ll_val <- coef(summary(lmm_ll))["(Intercept)", "Estimate"]
        t_stat_ll <- coef(summary(lmm_ll))["(Intercept)", "t value"]
    }
    
    final_results <- rbind(final_results, data.frame(
        Sigma_Pre = s_pre,
        Sigma_Post = s_post,
        Beta_RT = beta_rt_val,
        Beta_LL_Diff = beta_ll_val,
        T_Stat_LL = t_stat_ll
    ))
}

final_results <- final_results %>% arrange(desc(T_Stat_LL))
write_csv(final_results, "ar_event_weighted_grid_results.csv")
cat("\n==========================================\n")
cat("AR EVENT-WEIGHTED OPTIMIZATION RESULTS\n")
cat("==========================================\n")
print(final_results)
