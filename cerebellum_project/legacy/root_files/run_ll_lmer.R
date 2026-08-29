pacman::p_load(tidyverse, Rcpp, optimx, lme4, emmeans, parallel)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_dist_fatigue.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_v3.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 
best_N <- 20
num_cores <- detectCores() - 1

cat("Fitting Models and Extracting Trial-by-Trial Log-Likelihoods...\n")

results_final <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    iti <- d$ITI
    f_dur <- d$F_dur
    ttp <- d$ttp
    
    # 1. Dynamic QL Poly
    obj_dyn_poly <- function(phi) { return(eval_ql_ddm_dynamic_poly(phi, resp, out, rt) + lambda * sum(abs(phi))) }
    res_dyn_poly <- optim(rep(0, 8), obj_dyn_poly, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
    ll_dyn_poly <- extract_ll_ql_dynamic_poly_point(res_dyn_poly$par, resp, out, rt)
    
    # 2. Full Gating Fatigue N=20 (Phase 9)
    obj_fg_fatigue <- function(phi) { return(eval_bvk_full_gating_dist_fatigue(phi, resp, out, rt, iti, f_dur, best_N) + lambda * sum(abs(phi))) }
    res_fg_fatigue <- optim(rep(0, 11), obj_fg_fatigue, method="L-BFGS-B", lower=rep(-5, 11), upper=rep(5, 11))
    ll_fg_fatigue <- extract_ll_bvk_full_gating_dist_fatigue(res_fg_fatigue$par, resp, out, rt, iti, f_dur, best_N)
    
    # 3. Full Gating V3
    obj_fg_v3 <- function(phi) { return(eval_bvk_full_gating_v3(phi, resp, out, rt, iti, f_dur, ttp, best_N) + lambda * sum(abs(phi))) }
    res_fg_v3 <- optim(rep(0, 12), obj_fg_v3, method="L-BFGS-B", lower=rep(-5, 12), upper=rep(5, 12))
    ll_fg_v3 <- extract_ll_bvk_full_gating_v3(res_fg_v3$par, resp, out, rt, iti, f_dur, ttp, best_N)
    
    df1 <- data.frame(Participant = d$participant_id, Trial = d$ttp, LogLik = ll_dyn_poly, Model = "Dynamic QL Poly")
    df2 <- data.frame(Participant = d$participant_id, Trial = d$ttp, LogLik = ll_fg_fatigue, Model = paste0("Full Gating Fatigue (N=", best_N, ")"))
    df3 <- data.frame(Participant = d$participant_id, Trial = d$ttp, LogLik = ll_fg_v3, Model = paste0("Full Gating V3 (N=", best_N, ")"))
    
    return(bind_rows(df1, df2, df3))
}, mc.cores = num_cores)

res_df <- bind_rows(results_final)
res_df <- res_df %>% filter(!is.na(LogLik) & !is.infinite(LogLik))

res_df$Model <- factor(res_df$Model, levels=c("Dynamic QL Poly", paste0("Full Gating Fatigue (N=", best_N, ")"), paste0("Full Gating V3 (N=", best_N, ")")))

cat("\n==========================================\n")
cat("LINEAR MIXED MODEL (LMM) ON LOG-LIKELIHOODS\n")
cat("==========================================\n")
lmm_ll <- lmer(LogLik ~ Model + (1 | Participant), data = res_df, REML = FALSE)
print(summary(lmm_ll))

cat("\n==========================================\n")
cat("EMMEANS STATISTICAL COMPARISON\n")
cat("==========================================\n")
trends <- emmeans(lmm_ll, pairwise ~ Model)
print(trends)

cat("\n==========================================\n")
cat("OVERALL SUM LOG-LIKELIHOOD PER MODEL\n")
cat("==========================================\n")
res_df %>% group_by(Model) %>% summarize(Total_LL = sum(LogLik)) %>% print()
