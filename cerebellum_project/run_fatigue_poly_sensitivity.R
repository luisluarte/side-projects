pacman::p_load(tidyverse, Rcpp, optimx, lme4, emmeans, ggplot2, parallel)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_dist_fatigue_poly.cpp")
Rcpp::sourceCpp("src/models/extract_predicted_rt.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 
num_cores <- detectCores() - 1

N_MF_list <- c(5, 10, 20, 40, 80, 160)

cat("Running Grid Search over N_MF for Full Gating Fatigue Poly...\n")

grid_search_res <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    iti <- d$ITI
    f_dur <- d$F_dur
    
    # 1. Base QL Poly Model
    obj_dyn_poly <- function(phi) { return(eval_ql_ddm_dynamic_poly(phi, resp, out, rt) + lambda * sum(abs(phi))) }
    res_dyn_poly <- optim(rep(0, 8), obj_dyn_poly, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
    pred_rt_dyn_poly <- extract_rt_ql_dynamic_poly(res_dyn_poly$par, resp, out, rt)
    
    ql_df <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, 
                        Model = "Dynamic QL Poly", RT_prediction = pred_rt_dyn_poly)
    
    best_ll <- Inf
    best_N <- 5
    best_pred <- NULL
    
    # 2. Grid Search over N_MF
    res_list <- list()
    for (N in N_MF_list) {
        obj_fg <- function(phi) { return(eval_bvk_full_gating_dist_fatigue_poly(phi, resp, out, rt, iti, f_dur, N) + lambda * sum(abs(phi))) }
        res_fg <- optim(rep(0, 12), obj_fg, method="L-BFGS-B", lower=rep(-5, 12), upper=rep(5, 12))
        
        res_list[[as.character(N)]] <- res_fg$value
        
        if (res_fg$value < best_ll) {
            best_ll <- res_fg$value
            best_N <- N
            best_pred <- extract_rt_bvk_full_gating_dist_fatigue_poly(res_fg$par, resp, out, rt, iti, f_dur, N)
        }
    }
    
    best_fg_df <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, 
                             Model = "Full Gating Fatigue Poly", RT_prediction = best_pred)
    
    return(list(ql_df = ql_df, fg_df = best_fg_df, ll_list = res_list, best_N = best_N))
}, mc.cores = num_cores)

# Aggregate Grid Search
total_ll <- numeric(length(N_MF_list))
names(total_ll) <- as.character(N_MF_list)
for (res in grid_search_res) {
    for (N in names(total_ll)) {
        total_ll[N] <- total_ll[N] + res$ll_list[[N]]
    }
}
cat("\nTotal Deviance (-2*LL) per N_MF:\n")
print(total_ll)
optimal_N <- as.numeric(names(which.min(total_ll)))
cat("Optimal N_MF discovered:", optimal_N, "\n")

# To keep the comparison exact, we will re-extract the predictions using the GLOBAL optimal N for ALL subjects
# (Since the grid search returned the per-subject optimal, we need to standardize)
cat("\nRe-extracting predictions using GLOBAL optimal N =", optimal_N, "...\n")

final_df <- data.frame()
results_standardized <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    iti <- d$ITI
    f_dur <- d$F_dur
    
    # Base QL
    obj_dyn_poly <- function(phi) { return(eval_ql_ddm_dynamic_poly(phi, resp, out, rt) + lambda * sum(abs(phi))) }
    res_dyn_poly <- optim(rep(0, 8), obj_dyn_poly, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
    pred_rt_dyn_poly <- extract_rt_ql_dynamic_poly(res_dyn_poly$par, resp, out, rt)
    
    # FG with GLOBAL optimal N
    obj_fg <- function(phi) { return(eval_bvk_full_gating_dist_fatigue_poly(phi, resp, out, rt, iti, f_dur, optimal_N) + lambda * sum(abs(phi))) }
    res_fg <- optim(rep(0, 12), obj_fg, method="L-BFGS-B", lower=rep(-5, 12), upper=rep(5, 12))
    pred_rt_fg <- extract_rt_bvk_full_gating_dist_fatigue_poly(res_fg$par, resp, out, rt, iti, f_dur, optimal_N)
    
    df1 <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Dynamic QL Poly", RT_prediction = pred_rt_dyn_poly)
    df2 <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = paste0("Full Gating Fatigue Poly (N=", optimal_N, ")"), RT_prediction = pred_rt_fg)
    
    return(bind_rows(df1, df2))
}, mc.cores = num_cores)

for (df in results_standardized) final_df <- bind_rows(final_df, df)
final_df <- final_df %>% filter(!is.na(RT_prediction) & !is.infinite(RT_prediction) & RT_prediction > 0 & RT_prediction < 10)

cat("\n==========================================\n")
cat("PEARSON & SPEARMAN CORRELATIONS\n")
cat("==========================================\n")
for(m in unique(final_df$Model)) {
  m_df <- final_df %>% filter(Model == m)
  cat(m, "-> Pearson:", round(cor(m_df$RT_prediction, m_df$RT_empirical, method="pearson"), 4),
      "| Spearman:", round(cor(m_df$RT_prediction, m_df$RT_empirical, method="spearman"), 4), "\n")
}

cat("\n==========================================\n")
cat("LINEAR MIXED MODEL (LMM) ANALYSIS\n")
cat("==========================================\n")
final_df$Model <- factor(final_df$Model, levels=c("Dynamic QL Poly", paste0("Full Gating Fatigue Poly (N=", optimal_N, ")")))
final_df$RT_prediction_c <- scale(final_df$RT_prediction, center=TRUE, scale=FALSE)

lmm <- lmer(RT_empirical ~ Model * RT_prediction_c + (1 | Participant), data = final_df, REML = FALSE)
print(summary(lmm))

cat("\n==========================================\n")
cat("EMMEANS STATISTICAL COMPARISON (TRENDS)\n")
cat("==========================================\n")
trends <- emtrends(lmm, pairwise ~ Model, var="RT_prediction_c")
print(trends)

final_df <- final_df %>% mutate(log_RT_emp = log(RT_empirical), log_RT_pred = log(RT_prediction))

decile_df <- final_df %>%
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
  labs(title = paste0("Vincentized Calibration: Fatigue Poly (N=", optimal_N, ") vs QL Poly"),
       x = "Empirical log(RT) (Decile Means)",
       y = "Predicted log(RT) (Decile Means)",
       color = "Model") +
  theme_minimal() +
  scale_color_manual(values = setNames(c("#D55E00", "#CC79A7"), c("Dynamic QL Poly", paste0("Full Gating Fatigue Poly (N=", optimal_N, ")")))) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

ggsave("fatigue_poly_rt_decile_plot.pdf", p, width = 8, height = 6)
ggsave("fatigue_poly_rt_decile_plot.png", p, width = 8, height = 6, dpi=300)
