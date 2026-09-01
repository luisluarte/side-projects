pacman::p_load(tidyverse, Rcpp, cmaes)
Rcpp::sourceCpp("src/models/epoch4_champion_lti.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))

lambda_1 <- 1.0
lambda_2 <- 5.0
lambda_reg <- 0.05

fileConn <- file("magi_ledger.md", "a")
writeLines("\n### CMA-ES Optimization (Champion Model LTI)\n*   **Balthazar:** Nelder-Mead simplex is susceptible to local topological traps on stochastic generative landscapes. Escalating to Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to guarantee global exploration of the multi-objective surface.\n", fileConn)
close(fileConn)

cat("Evaluating LTI Cascade Champion Model with CMA-ES Evolutionary Strategy...\n")

cand_eval <- lapply(1:S, function(s_idx) {
    cat(sprintf("Running CMA-ES for Subject %d...\n", s_idx))
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) {
        rt_sim <- extract_epoch4_lti(phi, resp, out, rt)
        w1 <- mean(abs(sort(rt_sim) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
        if(is.na(beta_sim)) beta_sim <- 0
        return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
    }
    
    res <- cma_es(rep(0, 10), obj, lower=rep(-5, 10), upper=rep(5, 10), control=list(maxit=200))
    
    rt_pred <- extract_epoch4_lti(res$par, resp, out, rt)
    w1 <- mean(abs(sort(rt_pred) - sort(rt)))
    beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
    
    cat(sprintf("  -> Subj %d CMA-ES Done: W1=%.4f, Beta=%.4f\n", s_idx, w1, beta_sim))
    list(w1 = w1, beta = beta_sim, rt_pred = rt_pred, trials = nrow(d), par=res$par)
})

total_cand_w1 <- sum(unlist(lapply(cand_eval, function(x) x$w1))) / S
total_cand_beta <- mean(unlist(lapply(cand_eval, function(x) x$beta)), na.rm=TRUE)

cat(sprintf("\n[FINAL CMA-ES RESULT] LTI Champion: W1_Cand=%.4f, Beta_Cand=%.4f\n", total_cand_w1, total_cand_beta))

fileConn <- file("magi_ledger.md", "a")
writeLines(sprintf("*   **CMA-ES Result:** `W1_Cand=%.4f`, `Beta_Cand=%.4f`", total_cand_w1, total_cand_beta), fileConn)
close(fileConn)
cat("CMA-ES Epoch Complete.\n")
