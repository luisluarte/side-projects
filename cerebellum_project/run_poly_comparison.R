pacman::p_load(tidyverse, Rcpp, optimx, lme4, emmeans, ggplot2)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating.cpp")
Rcpp::sourceCpp("src/models/extract_predicted_rt.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 

res_df <- data.frame()
ll_dyn <- c()
ll_fg <- c()
ll_dyn_poly <- c()
ll_fg_poly <- c()

cat("Fitting Models and Extracting Predicted RTs (4 Models)...\n")
for(s_idx in 1:S) {
  d <- dat_clean %>% filter(participant_idx == s_idx)
  resp <- d$Boundary + 1
  out <- d$`F`
  rt <- d$RT
  iti <- d$ITI
  f_dur <- d$F_dur
  
  # 1. Dynamic QL
  obj_dyn <- function(phi) { return(eval_ql_ddm_dynamic(phi, resp, out, rt) + lambda * sum(abs(phi))) }
  res_dyn <- optim(rep(0, 7), obj_dyn, method="L-BFGS-B", lower=rep(-5, 7), upper=rep(5, 7))
  ll_dyn <- c(ll_dyn, extract_ll_ql_dynamic_point(res_dyn$par, resp, out, rt))
  pred_rt_dyn <- extract_rt_ql_dynamic(res_dyn$par, resp, out, rt)
  
  # 2. Full Gating
  obj_fg <- function(phi) { return(eval_bvk_full_gating(phi, resp, out, rt, iti, f_dur) + lambda * sum(abs(phi))) }
  res_fg <- optim(rep(0, 11), obj_fg, method="L-BFGS-B", lower=rep(-5, 11), upper=rep(5, 11))
  ll_fg <- c(ll_fg, extract_ll_bvk_full_gating(res_fg$par, resp, out, rt, iti, f_dur))
  pred_rt_fg <- extract_rt_bvk_full_gating(res_fg$par, resp, out, rt, iti, f_dur)
  
  # 3. Dynamic QL Poly
  obj_dyn_poly <- function(phi) { return(eval_ql_ddm_dynamic_poly(phi, resp, out, rt) + lambda * sum(abs(phi))) }
  res_dyn_poly <- optim(rep(0, 8), obj_dyn_poly, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
  ll_dyn_poly <- c(ll_dyn_poly, extract_ll_ql_dynamic_poly_point(res_dyn_poly$par, resp, out, rt))
  pred_rt_dyn_poly <- extract_rt_ql_dynamic_poly(res_dyn_poly$par, resp, out, rt)
  
  # 4. Full Gating Poly
  obj_fg_poly <- function(phi) { return(eval_bvk_full_gating_poly(phi, resp, out, rt, iti, f_dur) + lambda * sum(abs(phi))) }
  res_fg_poly <- optim(rep(0, 12), obj_fg_poly, method="L-BFGS-B", lower=rep(-5, 12), upper=rep(5, 12))
  ll_fg_poly <- c(ll_fg_poly, extract_ll_bvk_full_gating_poly(res_fg_poly$par, resp, out, rt, iti, f_dur))
  pred_rt_fg_poly <- extract_rt_bvk_full_gating_poly(res_fg_poly$par, resp, out, rt, iti, f_dur)
  
  res_df <- bind_rows(res_df, 
    data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Dynamic QL", RT_prediction = pred_rt_dyn),
    data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Full Gating", RT_prediction = pred_rt_fg),
    data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Dynamic QL Poly", RT_prediction = pred_rt_dyn_poly),
    data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Full Gating Poly", RT_prediction = pred_rt_fg_poly)
  )
}

# Ensure valid RT predictions
res_df <- res_df %>% filter(!is.na(RT_prediction) & !is.infinite(RT_prediction) & RT_prediction > 0 & RT_prediction < 10)

# ----------------- Log Likelihood Tests -----------------
N <- length(ll_dyn)
cat("\n==========================================\n")
cat("LOG-LIKELIHOOD COMPARISONS (AIC ADJUSTED)\n")
cat("==========================================\n")

cat("Dynamic QL Mean LL:", mean(ll_dyn), "(Sum:", sum(ll_dyn), ")\n")
cat("Dynamic QL Poly Mean LL:", mean(ll_dyn_poly), "(Sum:", sum(ll_dyn_poly), ")\n")
cat("Full Gating Mean LL:", mean(ll_fg), "(Sum:", sum(ll_fg), ")\n")
cat("Full Gating Poly Mean LL:", mean(ll_fg_poly), "(Sum:", sum(ll_fg_poly), ")\n\n")

# QL vs QL Poly
t_ql <- t.test((ll_dyn_poly - ll_dyn) - ((8*S - 7*S)/N), alternative="greater")
cat("Dynamic QL Poly vs Dynamic QL -> t:", t_ql$statistic, "p:", t_ql$p.value, "\n")

# FG vs FG Poly
t_fg <- t.test((ll_fg_poly - ll_fg) - ((12*S - 11*S)/N), alternative="greater")
cat("Full Gating Poly vs Full Gating -> t:", t_fg$statistic, "p:", t_fg$p.value, "\n")

# QL Poly vs FG Poly
t_vs <- t.test((ll_dyn_poly - ll_fg_poly) - ((8*S - 12*S)/N), alternative="greater")
cat("Dynamic QL Poly vs Full Gating Poly -> t:", t_vs$statistic, "p:", t_vs$p.value, "\n")

# ----------------- Correlations -----------------
cat("\n==========================================\n")
cat("PEARSON & SPEARMAN CORRELATIONS\n")
cat("==========================================\n")
for(m in unique(res_df$Model)) {
  m_df <- res_df %>% filter(Model == m)
  cat(m, "-> Pearson:", round(cor(m_df$RT_prediction, m_df$RT_empirical, method="pearson"), 4),
      "| Spearman:", round(cor(m_df$RT_prediction, m_df$RT_empirical, method="spearman"), 4), "\n")
}

# ----------------- Linear Mixed Model -----------------
cat("\n==========================================\n")
cat("LINEAR MIXED MODEL (LMM) ANALYSIS\n")
cat("==========================================\n")
res_df$Model <- factor(res_df$Model, levels=c("Dynamic QL", "Full Gating", "Dynamic QL Poly", "Full Gating Poly"))
res_df$RT_prediction_c <- scale(res_df$RT_prediction, center=TRUE, scale=FALSE)

lmm <- lmer(RT_empirical ~ Model * RT_prediction_c + (1 | Participant), data = res_df, REML = FALSE)
print(summary(lmm))

cat("\n==========================================\n")
cat("EMMEANS STATISTICAL COMPARISON (TRENDS)\n")
cat("==========================================\n")
trends <- emtrends(lmm, pairwise ~ Model, var="RT_prediction_c")
print(trends)

# ----------------- Plot -----------------
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
  labs(title = "Vincentized Calibration: Empirical vs. Predicted Log Reaction Time (Power Law)",
       x = "Empirical log(RT) (Decile Means)",
       y = "Predicted log(RT) (Decile Means)",
       color = "Model") +
  theme_minimal() +
  scale_color_manual(values = c("Dynamic QL" = "#E69F00", "Full Gating" = "#56B4E9",
                                "Dynamic QL Poly" = "#D55E00", "Full Gating Poly" = "#0072B2")) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

ggsave("poly_rt_decile_plot.pdf", p, width = 8, height = 6)
ggsave("poly_rt_decile_plot.png", p, width = 8, height = 6, dpi=300)

