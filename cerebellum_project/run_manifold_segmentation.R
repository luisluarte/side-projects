pacman::p_load(tidyverse, Rcpp, optimx, lme4, lmerTest, emmeans, parallel)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_dist_fatigue.cpp")
Rcpp::sourceCpp("src/models/extract_predicted_rt.cpp")

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

cat("Phase 1: Optimizing Models & Extracting Metrics on", num_cores, "cores...\n")

results_list <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    iti <- d$ITI
    f_dur <- d$F_dur
    T_trials <- nrow(d)
    
    # Global Unweighted Optimization
    obj_ql <- function(phi) { return(eval_ql_ddm_dynamic_poly(phi, resp, out, rt) + lambda * sum(abs(phi))) }
    res_ql <- optim(rep(0, 8), obj_ql, method="L-BFGS-B", lower=rep(-5, 8), upper=rep(5, 8))
    ll_ql <- extract_ll_ql_dynamic_poly_point(res_ql$par, resp, out, rt)
    rt_ql <- extract_rt_ql_dynamic_poly(res_ql$par, resp, out, rt)
    
    obj_fg <- function(phi) { return(eval_bvk_full_gating_dist_fatigue(phi, resp, out, rt, iti, f_dur, best_N) + lambda * sum(abs(phi))) }
    res_fg <- optim(rep(0, 11), obj_fg, method="L-BFGS-B", lower=rep(-5, 11), upper=rep(5, 11))
    ll_fg <- extract_ll_bvk_full_gating_dist_fatigue(res_fg$par, resp, out, rt, iti, f_dur, best_N)
    rt_fg <- extract_rt_bvk_full_gating_dist_fatigue(res_fg$par, resp, out, rt, iti, f_dur, best_N)
    
    df <- data.frame(
        Participant_ID = d$participant_id,
        Trial = 1:T_trials,
        buena = d$buena,
        Choice = d$Boundary,
        RT_emp = rt,
        RT_ql = rt_ql,
        RT_fg = rt_fg,
        LL_ql = ll_ql,
        LL_fg = ll_fg
    )
    return(df)
}, mc.cores = num_cores)

df_all <- bind_rows(results_list)

cat("Phase 2: Trial Tagging (Categorical Segmentation)...\n")
df_all <- df_all %>%
    group_by(Participant_ID) %>%
    mutate(
        Is_Reversal = (Trial > 1 & buena != lag(buena)),
        R_idx = cummax(ifelse(Is_Reversal, Trial, 0)),
        t_since_R = ifelse(R_idx > 0, Trial - R_idx, -1),
        is_switch = (Trial > 1 & Choice != lag(Choice))
    ) %>%
    group_by(Participant_ID, R_idx) %>%
    mutate(
        has_switched = cumsum(is_switch)
    ) %>%
    ungroup() %>%
    mutate(
        Cognitive_State = case_when(
            R_idx == 0 ~ "E_Unclassified",
            t_since_R > 5 & !is_switch ~ "State_A_Asymptotic_Exploitation",
            t_since_R > 5 & is_switch ~ "State_B_Exploratory_Deviation",
            t_since_R >= 1 & t_since_R <= 3 & !is_switch ~ "State_C_Environmental_Shock",
            is_switch & t_since_R >= 0 & has_switched == 1 ~ "State_D_Reactive_Shift",
            TRUE ~ "E_Unclassified"
        )
    )

df_all <- df_all %>%
    mutate(
        Delta_LL = LL_fg - LL_ql,
        Error_RT_ql = abs(RT_emp - RT_ql),
        Error_RT_fg = abs(RT_emp - RT_fg),
        Delta_Error_RT = Error_RT_ql - Error_RT_fg
    ) %>%
    filter(is.finite(Delta_LL) & is.finite(Delta_Error_RT))

cat("Phase 3: Mixed-Effects State Contrasts...\n")

df_filtered <- df_all %>% filter(Cognitive_State != "E_Unclassified")
df_filtered$Cognitive_State <- as.factor(df_filtered$Cognitive_State)

lmm_ll <- lmer(Delta_LL ~ 0 + Cognitive_State + (1 | Participant_ID), data = df_filtered)
lmm_rt <- lmer(Delta_Error_RT ~ 0 + Cognitive_State + (1 | Participant_ID), data = df_filtered)

summ_ll <- coef(summary(lmm_ll))
summ_rt <- coef(summary(lmm_rt))

res_table <- data.frame(
    State = gsub("Cognitive_State", "", rownames(summ_ll)),
    LL_Beta = summ_ll[, "Estimate"],
    LL_SE = summ_ll[, "Std. Error"],
    LL_t = summ_ll[, "t value"],
    RT_Beta = summ_rt[, "Estimate"],
    RT_SE = summ_rt[, "Std. Error"],
    RT_t = summ_rt[, "t value"]
)

write_csv(res_table, "manifold_segmentation_results.csv")
cat("\n==========================================\n")
cat("CATEGORICAL MANIFOLD SEGMENTATION RESULTS\n")
cat("==========================================\n")
print(res_table, row.names = FALSE)
