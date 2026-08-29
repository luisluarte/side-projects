pacman::p_load(tidyverse, Rcpp, optimx, lme4, lmerTest, emmeans, ggplot2, parallel, gridExtra)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_dist_fatigue.cpp")
Rcpp::sourceCpp("src/models/extract_predicted_rt.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 
best_N <- 20
num_cores <- detectCores() - 1

cat("Starting Backward Trimming Strategy Shift Analysis on", num_cores, "cores...\n")

results_list <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    N <- nrow(d)
    
    if(N < 45) return(NULL) 
    
    T_seq <- seq(N, 40, by = -5)
    subj_res <- list()
    
    for (T_val in T_seq) {
        d_sub <- d[1:T_val, ]
        resp <- d_sub$Boundary + 1
        out <- d_sub$`F`
        rt <- d_sub$RT
        iti <- d_sub$ITI
        f_dur <- d_sub$F_dur
        
        # Base QL Poly (8 params)
        obj_ql <- function(phi) { return(eval_ql_ddm_dynamic_poly(phi, resp, out, rt) + lambda * sum(abs(phi))) }
        res_ql <- optim(rep(0, 8), obj_ql, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
        ll_ql <- sum(extract_ll_ql_dynamic_poly_point(res_ql$par, resp, out, rt))
        
        # Dual FG Fatigue (11 params)
        obj_fg <- function(phi) { return(eval_bvk_full_gating_dist_fatigue(phi, resp, out, rt, iti, f_dur, best_N) + lambda * sum(abs(phi))) }
        res_fg <- optim(rep(0, 11), obj_fg, method="L-BFGS-B", lower=rep(-5, 11), upper=rep(5, 11))
        ll_fg <- sum(extract_ll_bvk_full_gating_dist_fatigue(res_fg$par, resp, out, rt, iti, f_dur, best_N))
        
        delta_ll <- ll_fg - ll_ql
        
        # Extract RT predictions for Dual
        pred_rt <- extract_rt_bvk_full_gating_dist_fatigue(res_fg$par, resp, out, rt, iti, f_dur, best_N)
        
        # OLS Beta_RT
        valid_idx <- which(!is.na(pred_rt) & !is.infinite(pred_rt) & pred_rt > 0 & pred_rt < 10)
        if (length(valid_idx) > 10) {
            fit_lm <- lm(rt[valid_idx] ~ pred_rt[valid_idx])
            beta_rt <- coef(fit_lm)[2]
        } else {
            beta_rt <- NA
        }
        
        subj_res[[length(subj_res) + 1]] <- data.frame(
            Participant_ID = d$participant_id[1],
            Trials_Retained = T_val,
            Trials_Removed = N - T_val,
            Delta_LL = delta_ll,
            Beta_RT = beta_rt
        )
    }
    return(bind_rows(subj_res))
}, mc.cores = num_cores)

final_df <- bind_rows(results_list) %>% filter(!is.na(Beta_RT))

write_csv(final_df, "backward_trimming_results.csv")

cat("\n==========================================\n")
cat("LMM STATISTICAL TESTS\n")
cat("==========================================\n")

lmm_ll <- lmer(Delta_LL ~ Trials_Removed + (1 | Participant_ID), data = final_df)
cat("\n--- Delta_LL ~ Trials_Removed ---\n")
print(summary(lmm_ll))

lmm_beta <- lmer(Beta_RT ~ Trials_Removed + (1 | Participant_ID), data = final_df)
cat("\n--- Beta_RT ~ Trials_Removed ---\n")
print(summary(lmm_beta))

p1 <- ggplot(final_df, aes(x = Trials_Removed, y = Delta_LL)) +
    geom_smooth(method = "gam", formula = y ~ s(x, k=5), color = "blue", fill = "lightblue", alpha = 0.5) +
    labs(title = "Advantage of Dual Cognitive Model over Heuristic Base",
         x = "Trials Removed (from End of Block)",
         y = "Delta Log-Likelihood (Dual - Base)") +
    theme_minimal()

p2 <- ggplot(final_df, aes(x = Trials_Removed, y = Beta_RT)) +
    geom_hline(yintercept = 1.0, linetype = "dashed", color = "red") +
    geom_smooth(method = "gam", formula = y ~ s(x, k=5), color = "darkgreen", fill = "lightgreen", alpha = 0.5) +
    labs(title = "RT Calibration (Slope -> 1.0 is Perfect)",
         x = "Trials Removed (from End of Block)",
         y = "Beta (Empirical ~ Predicted RT)") +
    theme_minimal()

p_combined <- arrangeGrob(p1, p2, ncol = 1)
ggsave("trimming_analysis_plot.pdf", p_combined, width = 8, height = 10)
ggsave("trimming_analysis_plot.png", p_combined, width = 8, height = 10, dpi=300)
