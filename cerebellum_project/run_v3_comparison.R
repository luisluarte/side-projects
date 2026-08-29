pacman::p_load(tidyverse, Rcpp, optimx, lme4, emmeans, ggplot2, parallel)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_dist_fatigue.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_v3.cpp")
Rcpp::sourceCpp("src/models/extract_predicted_rt.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 
best_N <- 20
num_cores <- detectCores() - 1

cat("Fitting V3 Model vs Baselines...\n")
res_df <- data.frame()

results_final <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    iti <- d$ITI
    f_dur <- d$F_dur
    ttp <- d$ttp
    
    # 1. Dynamic QL Poly (Base)
    obj_dyn_poly <- function(phi) { return(eval_ql_ddm_dynamic_poly(phi, resp, out, rt) + lambda * sum(abs(phi))) }
    res_dyn_poly <- optim(rep(0, 8), obj_dyn_poly, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
    pred_rt_dyn_poly <- extract_rt_ql_dynamic_poly(res_dyn_poly$par, resp, out, rt)
    
    # 2. Full Gating Fatigue N=20 (Phase 9)
    obj_fg_fatigue <- function(phi) { return(eval_bvk_full_gating_dist_fatigue(phi, resp, out, rt, iti, f_dur, best_N) + lambda * sum(abs(phi))) }
    res_fg_fatigue <- optim(rep(0, 11), obj_fg_fatigue, method="L-BFGS-B", lower=rep(-5, 11), upper=rep(5, 11))
    pred_rt_fg_fatigue <- extract_rt_bvk_full_gating_dist_fatigue(res_fg_fatigue$par, resp, out, rt, iti, f_dur, best_N)
    
    # 3. Full Gating V3 (New Proposal)
    obj_fg_v3 <- function(phi) { return(eval_bvk_full_gating_v3(phi, resp, out, rt, iti, f_dur, ttp, best_N) + lambda * sum(abs(phi))) }
    res_fg_v3 <- optim(rep(0, 12), obj_fg_v3, method="L-BFGS-B", lower=rep(-5, 12), upper=rep(5, 12))
    pred_rt_fg_v3 <- extract_rt_bvk_full_gating_v3(res_fg_v3$par, resp, out, rt, iti, f_dur, ttp, best_N)
    
    df1 <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Dynamic QL Poly", RT_prediction = pred_rt_dyn_poly)
    df2 <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = paste0("Full Gating Fatigue (N=", best_N, ")"), RT_prediction = pred_rt_fg_fatigue)
    df3 <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = paste0("Full Gating V3 (N=", best_N, ")"), RT_prediction = pred_rt_fg_v3)
    
    return(bind_rows(df1, df2, df3))
}, mc.cores = num_cores)

for (df in results_final) {
    res_df <- bind_rows(res_df, df)
}

res_df <- res_df %>% filter(!is.na(RT_prediction) & !is.infinite(RT_prediction) & RT_prediction > 0 & RT_prediction < 10)

cat("\n==========================================\n")
cat("PEARSON & SPEARMAN CORRELATIONS\n")
cat("==========================================\n")
for(m in unique(res_df$Model)) {
  m_df <- res_df %>% filter(Model == m)
  cat(m, "-> Pearson:", round(cor(m_df$RT_prediction, m_df$RT_empirical, method="pearson"), 4),
      "| Spearman:", round(cor(m_df$RT_prediction, m_df$RT_empirical, method="spearman"), 4), "\n")
}

cat("\n==========================================\n")
cat("LINEAR MIXED MODEL (LMM) ANALYSIS\n")
cat("==========================================\n")
res_df$Model <- factor(res_df$Model, levels=c("Dynamic QL Poly", paste0("Full Gating Fatigue (N=", best_N, ")"), paste0("Full Gating V3 (N=", best_N, ")")))
res_df$RT_prediction_c <- scale(res_df$RT_prediction, center=TRUE, scale=FALSE)

lmm <- lmer(RT_empirical ~ Model * RT_prediction_c + (1 | Participant), data = res_df, REML = FALSE)
print(summary(lmm))

cat("\n==========================================\n")
cat("EMMEANS STATISTICAL COMPARISON (TRENDS)\n")
cat("==========================================\n")
trends <- emtrends(lmm, pairwise ~ Model, var="RT_prediction_c")
print(trends)

res_df <- res_df %>% mutate(log_RT_emp = log(RT_empirical), log_RT_pred = log(RT_prediction))

decile_df <- res_df %>%
  group_by(Model, Participant) %>%
  mutate(decile = ntile(log_RT_emp, 10)) %>%
  group_by(Model, Participant, decile) %>%
  summarize(mean_log_emp = mean(log_RT_emp, na.rm = TRUE),
            mean_log_pred = mean(log_RT_pred, na.rm = TRUE),
            .groups = "drop")

p <- ggplot(decile_df, aes(x = mean_log_emp, y = mean_log_pred, color = Model, group = Model)) +
  geom_point(alpha = 0.3, size = 1.5) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  labs(title = paste0("Vincentized Calibration: V3 vs Baselines"),
       x = "Empirical log(RT) (Decile Means)",
       y = "Predicted log(RT) (Decile Means)",
       color = "Model") +
  theme_minimal() +
  scale_color_manual(values = setNames(c("#D55E00", "#009E73", "#56B4E9"), c("Dynamic QL Poly", paste0("Full Gating Fatigue (N=", best_N, ")"), paste0("Full Gating V3 (N=", best_N, ")")))) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

ggsave("v3_rt_decile_plot.pdf", p, width = 8, height = 6)
ggsave("v3_rt_decile_plot.png", p, width = 8, height = 6, dpi=300)
