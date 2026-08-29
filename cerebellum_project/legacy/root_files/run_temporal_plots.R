pacman::p_load(tidyverse, Rcpp, optimx, lme4, ggplot2, parallel, gridExtra, mgcv)

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

cat("Optimizing models to extract Temporal Dynamics on", num_cores, "cores...\n")

results_list <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    iti <- d$ITI
    f_dur <- d$F_dur
    
    # 1. Base QL Poly (8 params)
    obj_ql <- function(phi) { return(eval_ql_ddm_dynamic_poly(phi, resp, out, rt) + lambda * sum(abs(phi))) }
    res_ql <- optim(rep(0, 8), obj_ql, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
    ll_ql <- extract_ll_ql_dynamic_poly_point(res_ql$par, resp, out, rt)
    rt_ql <- extract_rt_ql_dynamic_poly(res_ql$par, resp, out, rt)
    
    # 2. Dual FG Fatigue (11 params)
    obj_fg <- function(phi) { return(eval_bvk_full_gating_dist_fatigue(phi, resp, out, rt, iti, f_dur, best_N) + lambda * sum(abs(phi))) }
    res_fg <- optim(rep(0, 11), obj_fg, method="L-BFGS-B", lower=rep(-5, 11), upper=rep(5, 11))
    ll_fg <- extract_ll_bvk_full_gating_dist_fatigue(res_fg$par, resp, out, rt, iti, f_dur, best_N)
    rt_fg <- extract_rt_bvk_full_gating_dist_fatigue(res_fg$par, resp, out, rt, iti, f_dur, best_N)
    
    df_ql <- data.frame(Participant = d$participant_id, Trial_Index = 1:nrow(d), 
                        RT_emp = rt, RT_pred = rt_ql, LogLik = ll_ql, Model = "Dynamic QL Poly")
    df_fg <- data.frame(Participant = d$participant_id, Trial_Index = 1:nrow(d), 
                        RT_emp = rt, RT_pred = rt_fg, LogLik = ll_fg, Model = "Full Gating Fatigue")
    
    return(bind_rows(df_ql, df_fg))
}, mc.cores = num_cores)

full_df <- bind_rows(results_list)

full_df <- full_df %>% 
    filter(!is.na(RT_pred) & !is.infinite(RT_pred) & RT_pred > 0 & RT_pred < 10) %>%
    filter(!is.na(LogLik) & !is.infinite(LogLik))

rmse_subj <- full_df %>% 
    group_by(Model, Participant) %>% 
    mutate(Decile = ntile(Trial_Index, 10)) %>% 
    group_by(Model, Participant, Decile) %>% 
    summarize(RMSE = sqrt(mean((RT_emp - RT_pred)^2, na.rm=TRUE)), .groups="drop")

rmse_agg <- rmse_subj %>% 
    group_by(Model, Decile) %>% 
    summarize(mean_RMSE = mean(RMSE), 
              se_RMSE = sd(RMSE)/sqrt(n()), .groups="drop")

p1 <- ggplot(rmse_agg, aes(x = Decile, y = mean_RMSE, color = Model, group = Model)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = mean_RMSE - se_RMSE, ymax = mean_RMSE + se_RMSE), width = 0.2, linewidth = 0.8) +
    scale_x_continuous(breaks = 1:10) +
    labs(title = "RT Predictive Error (RMSE) across Trial Deciles",
         subtitle = "Lower RMSE = Better linear prediction",
         x = "Trial Decile (Time ->)",
         y = "Mean RT RMSE (sec)") +
    theme_minimal() +
    scale_color_manual(values = c("Dynamic QL Poly" = "#D55E00", "Full Gating Fatigue" = "#009E73")) +
    theme(plot.title = element_text(face = "bold", size = 14), legend.position = "bottom")

p2 <- ggplot(full_df, aes(x = Trial_Index, y = LogLik, color = Model, fill = Model)) +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), linewidth = 1.2, alpha = 0.3) +
    labs(title = "Continuous Log-Likelihood Trajectory",
         subtitle = "Higher LogLik = Better density fit (GAM smoothed)",
         x = "Trial Index",
         y = "Log-Likelihood (per trial)") +
    theme_minimal() +
    scale_color_manual(values = c("Dynamic QL Poly" = "#D55E00", "Full Gating Fatigue" = "#009E73")) +
    scale_fill_manual(values = c("Dynamic QL Poly" = "#D55E00", "Full Gating Fatigue" = "#009E73")) +
    theme(plot.title = element_text(face = "bold", size = 14), legend.position = "bottom")

p_combined <- arrangeGrob(p1, p2, ncol = 1)
ggsave("temporal_dynamics_plot.pdf", p_combined, width = 8, height = 10)
ggsave("temporal_dynamics_plot.png", p_combined, width = 8, height = 10, dpi=300)

cat("\nPlotting complete. Saved to temporal_dynamics_plot.png\n")
