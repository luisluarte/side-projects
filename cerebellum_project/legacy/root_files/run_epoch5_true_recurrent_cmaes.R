pacman::p_load(tidyverse, Rcpp, cmaes)
Rcpp::sourceCpp("src/models/epoch5_true_recurrent.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))

# Restrict to 30 subjects for speed
S <- 30
dat_clean <- dat_clean %>% filter(participant_idx <= S)

lambda_1 <- 1.0
lambda_2 <- 5.0
lambda_reg <- 0.05

fileConn <- file("magi_ledger.md", "a")
writeLines("\n## Epoch: 5 (True Spatial Recurrent Topology) | 2026-08-25\n\n### 1. MAGI Consensus (The Recurrent Biological Truth)\n*   **Caspar:** We have established that no analytical trick (Uncoupled noise, AR(1) temporal reverberation, or LTI cascades) can bypass the Geometry/Sequence Pareto boundary. We now evaluate the true biological architecture: $O(N^2)$ Spatial Recurrence ($\\mathbf{W}_{res}$) mapped over independent biological stochasticity ($\\mathbf{\\zeta}$).\n*   **Balthazar:** CMA-ES evolutionary solver engaged over the multi-objective loss. $N=30$ subjects to respect the computational weight of full matrices.\n\n### 2. Epoch Results Summary\n", fileConn)
close(fileConn)

cat("Fitting Baseline on 30 subjects...\n")
base_eval <- lapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) { 
        rt_sim <- extract_baseline_exgauss_sim(phi, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    res <- cma_es(rep(0, 4), obj, lower=rep(-5, 4), upper=rep(5, 4), control=list(maxit=100))
    rt_pred <- extract_baseline_exgauss_sim(res$par, resp, out, rt)
    w1 <- mean(abs(sort(rt_pred) - sort(rt)))
    beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
    list(w1 = w1, beta = beta_sim)
})
all_base_w1 <- sum(unlist(lapply(base_eval, function(x) x$w1))) / S
all_base_beta <- mean(unlist(lapply(base_eval, function(x) x$beta)), na.rm=TRUE)
cat(sprintf("Baseline N=30: W1=%.4f, Beta=%.4f\n\n", all_base_w1, all_base_beta))

cat("Evaluating True Spatial Recurrent Topology with CMA-ES...\n")
# Fixed hyperparameters for the test: L_min=0.01, L_max=1.00, Pois=5.0
hyper <- c(0.01, 1.0, 5.0) 

cand_eval <- lapply(1:S, function(s_idx) {
    cat(sprintf("Running CMA-ES for Recurrent Network (Subject %d)...\n", s_idx))
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) {
        rt_sim <- extract_epoch5_true_recurrent(phi, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    
    res <- cma_es(rep(0, 10), obj, lower=rep(-5, 10), upper=rep(5, 10), control=list(maxit=150))
    
    rt_pred <- extract_epoch5_true_recurrent(res$par, hyper, resp, out, rt)
    w1 <- mean(abs(sort(rt_pred) - sort(rt)))
    beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
    
    cat(sprintf("  -> Subj %d True-Recurrent Done: W1=%.4f, Beta=%.4f\n", s_idx, w1, beta_sim))
    list(w1 = w1, beta = beta_sim)
})

total_cand_w1 <- sum(unlist(lapply(cand_eval, function(x) x$w1))) / S
total_cand_beta <- mean(unlist(lapply(cand_eval, function(x) x$beta)), na.rm=TRUE)

cat(sprintf("\n[FINAL CMA-ES RESULT] TRUE RECURRENT MODEL: W1_Cand=%.4f (Base=%.4f), Beta_Cand=%.4f (Base=%.4f)\n", 
    total_cand_w1, all_base_w1, total_cand_beta, all_base_beta))

fileConn <- file("magi_ledger.md", "a")
writeLines(sprintf("*   **True Recurrent Result (N=30):** `W1_Cand=%.4f` (vs Base `%.4f`), `Beta_Cand=%.4f` (vs Base `%.4f`)", total_cand_w1, all_base_w1, total_cand_beta, all_base_beta), fileConn)
close(fileConn)
cat("Epoch 5 True Recurrent Complete.\n")
