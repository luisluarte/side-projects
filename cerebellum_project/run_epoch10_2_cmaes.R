pacman::p_load(tidyverse, Rcpp, cmaes)
Rcpp::sourceCpp("src/models/epoch10_2_wald_decoupled.cpp")

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

lambda_1 <- 1.0; lambda_2 <- 5.0; lambda_reg <- 0.05
hyper <- c(0.01, 1.00, 2.0) 

fileConn <- file("magi_ledger.md", "a")
writeLines("\n## Epoch: 10.2 (Hybrid Composition: Boundary Collapse & Decoupled Diffusion | Variant 11.2)\n*   **Objective:** Resolving the W1 bottleneck via Proposal 1 (Cerebellar boundary collapse) and Proposal 2 (Diffusion scale decoupling). Running 12-parameter CMA-ES optimization.\n", fileConn)
close(fileConn)

cand_eval <- lapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) {
        rt_sim <- extract_epoch10_2_hybrid(phi, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    
    res <- cma_es(rep(0, 12), obj, lower=rep(-5, 12), upper=rep(5, 12), control=list(maxit=150))
    rt_pred <- extract_epoch10_2_hybrid(res$par, hyper, resp, out, rt)
    w1 <- mean(abs(sort(rt_pred) - sort(rt)))
    beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
    
    list(w1 = w1, beta = beta_sim)
})

total_cand_w1 <- sum(unlist(lapply(cand_eval, function(x) x$w1))) / S
total_cand_beta <- mean(unlist(lapply(cand_eval, function(x) x$beta)), na.rm=TRUE)

cand_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
    d <- dat_noise %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) {
        rt_sim <- extract_epoch10_2_hybrid(phi, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    cma_es(rep(0, 12), obj, lower=rep(-5, 12), upper=rep(5, 12), control=list(maxit=100))$value
})
base_noise_scores <- c(0.40, 0.42, 0.39, 0.45, 0.41)
rad_diff <- unlist(cand_noise_eval) - base_noise_scores
t_test_rad <- t.test(rad_diff, alternative="less") 
rad_p <- t_test_rad$p.value 

fileConn <- file("magi_ledger.md", "a")
writeLines(sprintf("*   **Decoupled Wald Result (N=30):** `W1_Cand=%.4f`, `Beta_Cand=%.4f`, `Rad_p=%.3f`", total_cand_w1, total_cand_beta, rad_p), fileConn)
close(fileConn)
