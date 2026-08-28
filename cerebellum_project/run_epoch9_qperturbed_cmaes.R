pacman::p_load(tidyverse, Rcpp, cmaes)
Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- 30
dat_clean <- dat_clean %>% filter(participant_idx <= S)

set.seed(99)
dat_noise <- dat_clean %>% group_by(participant_idx) %>% mutate(`F` = sample(`F`)) %>% ungroup()
rademacher_subjects <- 1:5 

lambda_1 <- 1.0
lambda_2 <- 5.0
lambda_reg <- 0.05

hyper <- c(0.01, 1.00, 2.0) 

fileConn <- file("magi_ledger.md", "a")
writeLines("\n## Epoch: 9 (Q-Perturbed Fractional Wald Manifold | Variant 11)\n\n### 1. MAGI Consensus (The Cerebello-Thalamo-Cortical Loop)\n*   **Caspar:** Formalizing the Cerebellum not as a physical execution smoothing engine, but as a parallel expected-value predictor. The $1/f$ spectral manifold now generates continuous $Q_{cb}$ predictions to perturb the baseline Cortical Drift Rate ($v^{(t)}$).\n*   **Balthazar:** Generative evaluation relies entirely on the pure Inverse Gaussian (Wald). The Gaussian motor noise envelope has been aborted. Optimizing the 9-dimensional parameter space via CMA-ES.\n", fileConn)
close(fileConn)

cat("Evaluating Q-Perturbed Fractional Manifold (Variant 11) with CMA-ES...\n")
cat("Reference Baseline Wald (N=30): W1 = 0.2981, Beta = 0.4632\n")

# A. Evaluate Candidate on True Data
cand_eval <- lapply(1:S, function(s_idx) {
    cat(sprintf("  -> Subj %d...\n", s_idx))
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) {
        rt_sim <- extract_epoch9_qperturbed(phi, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    
    res <- cma_es(rep(0, 9), obj, lower=rep(-5, 9), upper=rep(5, 9), control=list(maxit=150))
    rt_pred <- extract_epoch9_qperturbed(res$par, hyper, resp, out, rt)
    w1 <- mean(abs(sort(rt_pred) - sort(rt)))
    beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
    
    list(w1 = w1, beta = beta_sim)
})

total_cand_w1 <- sum(unlist(lapply(cand_eval, function(x) x$w1))) / S
total_cand_beta <- mean(unlist(lapply(cand_eval, function(x) x$beta)), na.rm=TRUE)

# B. Rademacher Capacity Parity
cat("Evaluating Rademacher Parity...\n")
base_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
    d <- dat_noise %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) { 
        rt_sim <- extract_baseline_wald_sim(phi, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    optim(rep(0, 4), obj, method="Nelder-Mead", control=list(maxit=100))$value
})

cand_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
    d <- dat_noise %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) {
        rt_sim <- extract_epoch9_qperturbed(phi, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    cma_es(rep(0, 9), obj, lower=rep(-5, 9), upper=rep(5, 9), control=list(maxit=100))$value
})

rad_diff <- unlist(cand_noise_eval) - unlist(base_noise_eval)
t_test_rad <- t.test(rad_diff, alternative="less") 
rad_p <- t_test_rad$p.value 

cat(sprintf("\n[FINAL CMA-ES RESULT] Q-Perturbed Wald: W1_Cand=%.4f, Beta_Cand=%.4f, Rad_p=%.3f\n", 
    total_cand_w1, total_cand_beta, rad_p))

fileConn <- file("magi_ledger.md", "a")
writeLines(sprintf("*   **Q-Perturbed Wald Result (N=30):** `W1_Cand=%.4f`, `Beta_Cand=%.4f`, `Rad_p=%.3f`", total_cand_w1, total_cand_beta, rad_p), fileConn)
close(fileConn)
cat("Epoch 9 Q-Perturbed Complete.\n")
