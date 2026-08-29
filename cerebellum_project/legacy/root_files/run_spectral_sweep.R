pacman::p_load(tidyverse, zoo, lme4, lmerTest, TTR)

sim_data <- read_csv("results/tables/magi_raw_trial_simulations_N30.csv", show_col_types = FALSE)

T_seq <- 3:10
results <- data.frame()

for(T_val in T_seq) {
    
    d_filtered <- sim_data %>%
        group_by(SubjectID) %>%
        mutate(
            Emp_Box = rollmean(Empirical_RT, k=T_val, fill=NA, align="right"),
            Base_Box = rollmean(Baseline_Wald_Expected_RT, k=T_val, fill=NA, align="right"),
            Term_Box = rollmean(Terminal_Hybrid_Expected_RT, k=T_val, fill=NA, align="right"),
            
            Emp_EMA = EMA(Empirical_RT, n=T_val),
            Base_EMA = EMA(Baseline_Wald_Expected_RT, n=T_val),
            Term_EMA = EMA(Terminal_Hybrid_Expected_RT, n=T_val)
        ) %>%
        ungroup() %>%
        filter(!is.na(Emp_Box), !is.na(Emp_EMA))
        
    mod_base_box <- lmer(Emp_Box ~ Base_Box + (1|SubjectID), data=d_filtered)
    mod_term_box <- lmer(Emp_Box ~ Term_Box + (1|SubjectID), data=d_filtered)
    
    mod_base_ema <- lmer(Emp_EMA ~ Base_EMA + (1|SubjectID), data=d_filtered)
    mod_term_ema <- lmer(Emp_EMA ~ Term_EMA + (1|SubjectID), data=d_filtered)
    
    extract_stats <- function(mod) {
        coefs <- summary(mod)$coefficients
        beta <- coefs[2, "Estimate"]
        pval <- coefs[2, "Pr(>|t|)"]
        return(c(beta, pval))
    }
    
    s_bb <- extract_stats(mod_base_box)
    s_tb <- extract_stats(mod_term_box)
    s_be <- extract_stats(mod_base_ema)
    s_te <- extract_stats(mod_term_ema)
    
    results <- bind_rows(results, data.frame(
        T_Lag = T_val, Filter = "Boxcar", Model = "Baseline Wald", Beta = s_bb[1], P_Value = s_bb[2]
    ))
    results <- bind_rows(results, data.frame(
        T_Lag = T_val, Filter = "Boxcar", Model = "Terminal Hybrid", Beta = s_tb[1], P_Value = s_tb[2]
    ))
    results <- bind_rows(results, data.frame(
        T_Lag = T_val, Filter = "EMA", Model = "Baseline Wald", Beta = s_be[1], P_Value = s_be[2]
    ))
    results <- bind_rows(results, data.frame(
        T_Lag = T_val, Filter = "EMA", Model = "Terminal Hybrid", Beta = s_te[1], P_Value = s_te[2]
    ))
}

results_wide <- results %>%
    mutate(
        Beta_str = sprintf("%.4f", Beta),
        P_str = sprintf("%.4e", P_Value),
        Sig = case_when(P_Value < 0.001 ~ "***", P_Value < 0.01 ~ "**", P_Value < 0.05 ~ "*", TRUE ~ "")
    ) %>%
    select(Filter, T_Lag, Model, Beta_str, P_str, Sig) %>%
    arrange(Filter, T_Lag, Model)
    
write_csv(results_wide, "results/tables/spectral_sweep_results.csv")
print(as.data.frame(results_wide))
