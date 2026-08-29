pacman::p_load(tidyverse, pROC, PRROC, pracma)

sim_data <- read_csv("results/tables/magi_raw_trial_simulations_N30_CMAES.csv", show_col_types = FALSE)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
p_base <- read_csv("results/tables/magi_baseline_opt_params_CMAES.csv", show_col_types = FALSE)$Value
p_var <- read_csv("results/tables/magi_terminal_hybrid_opt_params_CMAES.csv", show_col_types = FALSE)$Value

alpha_base <- 1.0 / (1.0 + exp(-p_base[4]))
alpha_term <- 1.0 / (1.0 + exp(-p_var[4]))

dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0),
           Reward = `F`) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id))) %>% filter(participant_idx <= 30)

dat_clean <- dat_clean %>% group_by(participant_idx) %>%
    mutate(
        Is_Switch = ifelse(row_number() > 1 & Resp != lag(Resp), 1, 0),
        Prev_Resp = lag(Resp)
    ) %>% ungroup()

p_switch_base <- numeric(nrow(dat_clean))
p_switch_term <- numeric(nrow(dat_clean))

idx <- 1
for(s in 1:30) {
    d <- dat_clean %>% filter(participant_idx == s)
    n <- nrow(d)
    Q_b <- c(0.5, 0.5)
    Q_t <- c(0.5, 0.5)
    
    for(t in 1:n) {
        ch <- d$Boundary[t] + 1
        r <- d$Reward[t]
        
        if(t > 1) {
            prev_ch <- d$Prev_Resp[t]
            p_switch_base[idx] <- 1 - (Q_b[prev_ch] / (Q_b[1] + Q_b[2]))
            p_switch_term[idx] <- 1 - (Q_t[prev_ch] / (Q_t[1] + Q_t[2]))
        } else {
            p_switch_base[idx] <- NA
            p_switch_term[idx] <- NA
        }
        
        Q_b[ch] <- Q_b[ch] + alpha_base * (r - Q_b[ch])
        Q_t[ch] <- Q_t[ch] + alpha_term * (r - Q_t[ch])
        idx <- idx + 1
    }
}

dat_clean$P_Switch_Base <- p_switch_base
dat_clean$P_Switch_Term <- p_switch_term
eval_df <- dat_clean %>% filter(!is.na(Is_Switch), !is.na(P_Switch_Base))

calc_class_metrics <- function(truth, probs) {
    roc_obj <- pROC::roc(truth, probs, quiet=TRUE)
    roc_auc <- as.numeric(pROC::auc(roc_obj))
    
    pr_obj <- PRROC::pr.curve(scores.class0 = probs[truth == 1], scores.class1 = probs[truth == 0], curve=FALSE)
    pr_auc <- pr_obj$auc.integral
    
    preds <- ifelse(probs >= 0.5, 1, 0)
    tp <- sum(preds == 1 & truth == 1)
    tn <- sum(preds == 0 & truth == 0)
    fp <- sum(preds == 1 & truth == 0)
    fn <- sum(preds == 0 & truth == 1)
    
    mcc_den <- sqrt(as.numeric(tp + fp) * as.numeric(tp + fn) * as.numeric(tn + fp) * as.numeric(tn + fn))
    mcc <- ifelse(mcc_den == 0, 0, ((tp * tn) - (fp * fn)) / mcc_den)
    
    precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    f1 <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
    
    return(c(roc_auc, pr_auc, mcc, f1))
}

m_base <- calc_class_metrics(eval_df$Is_Switch, eval_df$P_Switch_Base)
m_term <- calc_class_metrics(eval_df$Is_Switch, eval_df$P_Switch_Term)

cat("=== SWITCH PREDICTION METRICS (Positive Class = Switch) ===\n")
cat(sprintf("%-15s %-15s %-15s\n", "Metric", "Baseline Wald", "Terminal Hybrid"))
cat(sprintf("%-15s %-15.4f %-15.4f\n", "ROC-AUC", m_base[1], m_term[1]))
cat(sprintf("%-15s %-15.4f %-15.4f\n", "PR-AUC", m_base[2], m_term[2]))
cat(sprintf("%-15s %-15.4f %-15.4f\n", "MCC", m_base[3], m_term[3]))
cat(sprintf("%-15s %-15.4f %-15.4f\n", "F1 Score", m_base[4], m_term[4]))

cat("\n=== HURST EXPONENT (1/f Pink Noise) ===\n")
h_emp <- numeric(30)
h_base <- numeric(30)
h_term <- numeric(30)

for(s in 1:30) {
    ds <- sim_data %>% filter(SubjectID == s)
    h_emp[s] <- hurstexp(ds$Empirical_RT, display=FALSE)$Hs
    h_base[s] <- hurstexp(ds$Baseline_Wald_Sim_RT, display=FALSE)$Hs
    h_term[s] <- hurstexp(ds$Terminal_Hybrid_Sim_RT, display=FALSE)$Hs
}
cat(sprintf("Empirical Mean Hurst: %.4f\n", mean(h_emp)))
cat(sprintf("Baseline Mean Hurst:  %.4f\n", mean(h_base)))
cat(sprintf("Terminal Mean Hurst:  %.4f\n", mean(h_term)))

cat("\n=== DYNAMIC VARIANCE EXPANSION (Late vs Early StdDev) ===\n")
var_df <- sim_data %>%
    group_by(SubjectID) %>%
    mutate(Fatigue_Q = ntile(CumulativeFatigue, 5)) %>%
    filter(Fatigue_Q %in% c(1, 5)) %>%
    group_by(SubjectID, Fatigue_Q) %>%
    summarize(sd_emp = sd(Empirical_RT), sd_base = sd(Baseline_Wald_Sim_RT), sd_term = sd(Terminal_Hybrid_Sim_RT), .groups="drop") %>%
    pivot_wider(names_from = Fatigue_Q, values_from = c(sd_emp, sd_base, sd_term)) %>%
    mutate(delta_sd_emp = sd_emp_5 - sd_emp_1, delta_sd_base = sd_base_5 - sd_base_1, delta_sd_term = sd_term_5 - sd_term_1)

cat(sprintf("Empirical SD Delta: %.4f\n", mean(var_df$delta_sd_emp)))
cat(sprintf("Baseline SD Delta:  %.4f\n", mean(var_df$delta_sd_base)))
cat(sprintf("Terminal SD Delta:  %.4f\n", mean(var_df$delta_sd_term)))
