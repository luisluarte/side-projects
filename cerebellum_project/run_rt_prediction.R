pacman::p_load(tidyverse, Rcpp, optimx, lme4, emmeans)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating.cpp")
Rcpp::sourceCpp("src/models/extract_predicted_rt.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 

res_df <- data.frame()

cat("Fitting Models and Extracting Predicted RTs...\n")
for(s_idx in 1:S) {
  d <- dat_clean %>% filter(participant_idx == s_idx)
  resp <- d$Boundary + 1
  out <- d$`F`
  rt <- d$RT
  iti <- d$ITI
  f_dur <- d$F_dur
  
  obj_dyn <- function(phi) {
    dev <- eval_ql_ddm_dynamic(phi, resp, out, rt)
    return(dev + lambda * sum(abs(phi)))
  }
  init_phi_dyn <- rep(0, 7)
  res_dyn <- optim(init_phi_dyn, obj_dyn, method="L-BFGS-B", lower=rep(-5, 7), upper=rep(5, 7))
  pred_rt_dyn <- extract_rt_ql_dynamic(res_dyn$par, resp, out, rt)
  
  obj_fg <- function(phi) {
    dev <- eval_bvk_full_gating(phi, resp, out, rt, iti, f_dur)
    return(dev + lambda * sum(abs(phi)))
  }
  init_phi_fg <- rep(0, 11)
  res_fg <- optim(init_phi_fg, obj_fg, method="L-BFGS-B", lower=rep(-5, 11), upper=rep(5, 11))
  pred_rt_fg <- extract_rt_bvk_full_gating(res_fg$par, resp, out, rt, iti, f_dur)
  
  df_dyn <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Dynamic_QL", RT_prediction = pred_rt_dyn)
  df_fg <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Full_Gating", RT_prediction = pred_rt_fg)
  
  res_df <- bind_rows(res_df, df_dyn, df_fg)
}

res_df$Model <- as.factor(res_df$Model)

# Ensure no NaNs from the analytical prediction
res_df <- res_df %>% filter(!is.na(RT_prediction) & !is.infinite(RT_prediction) & RT_prediction > 0 & RT_prediction < 10)

cat("\n==========================================\n")
cat("PEARSON & SPEARMAN CORRELATIONS\n")
cat("==========================================\n")
cor_dyn_p <- cor(res_df$RT_prediction[res_df$Model == "Dynamic_QL"], res_df$RT_empirical[res_df$Model == "Dynamic_QL"], method="pearson")
cor_dyn_s <- cor(res_df$RT_prediction[res_df$Model == "Dynamic_QL"], res_df$RT_empirical[res_df$Model == "Dynamic_QL"], method="spearman")
cat("Dynamic QL  -> Pearson: ", round(cor_dyn_p, 4), " | Spearman: ", round(cor_dyn_s, 4), "\n")

cor_fg_p <- cor(res_df$RT_prediction[res_df$Model == "Full_Gating"], res_df$RT_empirical[res_df$Model == "Full_Gating"], method="pearson")
cor_fg_s <- cor(res_df$RT_prediction[res_df$Model == "Full_Gating"], res_df$RT_empirical[res_df$Model == "Full_Gating"], method="spearman")
cat("Full Gating -> Pearson: ", round(cor_fg_p, 4), " | Spearman: ", round(cor_fg_s, 4), "\n")

cat("\n==========================================\n")
cat("LINEAR MIXED MODEL (LMM) ANALYSIS\n")
cat("==========================================\n")
cat("Formula: RT_empirical ~ Model * RT_prediction + (1 | Participant)\n\n")

# Scale RT predictions to help LMM convergence
res_df$RT_prediction_c <- scale(res_df$RT_prediction, center=TRUE, scale=FALSE)
lmm <- lmer(RT_empirical ~ Model * RT_prediction_c + (1 | Participant), data = res_df, REML = FALSE)
print(summary(lmm))

cat("\n==========================================\n")
cat("EMMEANS STATISTICAL COMPARISON (TRENDS)\n")
cat("==========================================\n")
trends <- emtrends(lmm, pairwise ~ Model, var="RT_prediction_c")
print(trends)

