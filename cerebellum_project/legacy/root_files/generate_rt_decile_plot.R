pacman::p_load(tidyverse, Rcpp, optimx, ggplot2)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating.cpp")
Rcpp::sourceCpp("src/models/extract_predicted_rt.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 

res_df <- data.frame()

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
  
  df_dyn <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Dynamic QL", RT_prediction = pred_rt_dyn)
  df_fg <- data.frame(Participant = d$participant_id, Trial = d$ttp, RT_empirical = rt, Model = "Full Gating", RT_prediction = pred_rt_fg)
  
  res_df <- bind_rows(res_df, df_dyn, df_fg)
}

res_df <- res_df %>% filter(!is.na(RT_prediction) & !is.infinite(RT_prediction) & RT_prediction > 0 & RT_prediction < 10)

# 1. Compute logs
res_df <- res_df %>%
  mutate(log_RT_emp = log(RT_empirical),
         log_RT_pred = log(RT_prediction))

# 2. Divide into deciles per participant and model, compute means
decile_df <- res_df %>%
  group_by(Model, Participant) %>%
  mutate(decile = ntile(log_RT_emp, 10)) %>%
  group_by(Model, Participant, decile) %>%
  summarize(mean_log_emp = mean(log_RT_emp, na.rm = TRUE),
            mean_log_pred = mean(log_RT_pred, na.rm = TRUE),
            .groups = "drop")

# 3. Plot
p <- ggplot(decile_df, aes(x = mean_log_emp, y = mean_log_pred, color = Model, group = Model)) +
  geom_point(alpha = 0.3, size = 1.5) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  labs(title = "Vincentized Calibration: Empirical vs. Predicted Log Reaction Time",
       x = "Empirical log(RT) (Decile Means)",
       y = "Predicted log(RT) (Decile Means)",
       color = "Model") +
  theme_minimal() +
  scale_color_manual(values = c("Dynamic QL" = "#E69F00", "Full Gating" = "#56B4E9")) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

ggsave("rt_decile_plot.pdf", p, width = 8, height = 6)
ggsave("rt_decile_plot.png", p, width = 8, height = 6, dpi=300)

