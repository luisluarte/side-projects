pacman::p_load(tidyverse, Rcpp, cmaes)
Rcpp::sourceCpp("src/models/epoch7_corticopontine.cpp")
Rcpp::sourceCpp("src/models/epoch6_contextual_manifold.cpp") # for baseline_exgauss_sim

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- 30
dat_clean <- dat_clean %>% filter(participant_idx <= S)

# Rademacher Noise Setup
set.seed(99)
dat_noise <- dat_clean %>% group_by(participant_idx) %>% mutate(`F` = sample(`F`)) %>% ungroup()
rademacher_subjects <- 1:5 

lambda_1 <- 1.0
lambda_2 <- 5.0
lambda_reg <- 0.05

hyper <- c(0.01, 1.00, 2.0) 

fileConn <- file("magi_ledger.md", "a")
writeLines("\n## Epoch: 7 (Cortico-Pontine Deep-Time Integration | Variant 9.2)\n\n### 1. MAGI Consensus (The Dual-Scale Reservoir)\n*   **Caspar:** The Contextual Manifold successfully bounded generative geometry but sacrificed slow fatigue. By injecting both continuous cognitive Q-states and an EMA of volatility, the granular layer is now a true dual-scale reservoir.\n*   **Balthazar:** CMA-ES evolutionary matrix initializing (10 Parameters, including $\\gamma_{ema}$). Rademacher Parity Gate strictly enforced.\n", fileConn)
close(fileConn)

cat("Evaluating Cortico-Pontine Manifold (Variant 9.2) with CMA-ES...\n")
cat("Reference Baseline (N=30): W1 = 0.2418, Beta = 0.2723\n")
cat("Reference Contextual Var 9.1 (N=30): W1 = 0.3428, Beta = 0.0832\n")

# A. Evaluate the Candidate on True Data
cand_eval <- lapply(1:S, function(s_idx) {
    cat(sprintf("  -> Subj %d...\n", s_idx))
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) {
        rt_sim <- extract_epoch7_corticopontine(phi, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    
    res <- cma_es(rep(0, 10), obj, lower=rep(-5, 10), upper=rep(5, 10), control=list(maxit=150))
    rt_pred <- extract_epoch7_corticopontine(res$par, hyper, resp, out, rt)
    w1 <- mean(abs(sort(rt_pred) - sort(rt)))
    beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
    
    list(w1 = w1, beta = beta_sim)
})

total_cand_w1 <- sum(unlist(lapply(cand_eval, function(x) x$w1))) / S
total_cand_beta <- mean(unlist(lapply(cand_eval, function(x) x$beta)), na.rm=TRUE)

# B. Evaluate Rademacher Capacity
cat("Evaluating Rademacher Parity...\n")
# 1. Baseline Noise eval
base_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
    d <- dat_noise %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) { 
        rt_sim <- extract_baseline_exgauss_sim(phi, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    optim(rep(0, 4), obj, method="Nelder-Mead", control=list(maxit=100))$value
})

# 2. Candidate Noise eval
cand_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
    d <- dat_noise %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) {
        rt_sim <- extract_epoch7_corticopontine(phi, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    cma_es(rep(0, 10), obj, lower=rep(-5, 10), upper=rep(5, 10), control=list(maxit=100))$value
})

rad_diff <- unlist(cand_noise_eval) - unlist(base_noise_eval)
t_test_rad <- t.test(rad_diff, alternative="less") 
rad_p <- t_test_rad$p.value 

cat(sprintf("\n[FINAL CMA-ES RESULT] Cortico-Pontine Manifold: W1_Cand=%.4f, Beta_Cand=%.4f, Rad_p=%.3f\n", 
    total_cand_w1, total_cand_beta, rad_p))

fileConn <- file("magi_ledger.md", "a")
writeLines(sprintf("*   **Cortico-Pontine Result (N=30):** `W1_Cand=%.4f`, `Beta_Cand=%.4f`, `Rad_p=%.3f`", total_cand_w1, total_cand_beta, rad_p), fileConn)
close(fileConn)
cat("Epoch 7 Cortico-Pontine Complete.\n")
