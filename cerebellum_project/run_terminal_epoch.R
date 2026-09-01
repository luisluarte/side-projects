pacman::p_load(tidyverse, Rcpp)
Rcpp::sourceCpp("src/models/epoch2_wasserstein_landscape.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))

set.seed(99)
dat_noise <- dat_clean %>% group_by(participant_idx) %>% mutate(`F` = sample(`F`)) %>% ungroup()
rademacher_subjects <- 1:5 

lambda_1 <- 1.0
lambda_2 <- 5.0
lambda_reg <- 0.05

cat("Phase 2: Baseline Composite Fit...\n")
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
    res <- optim(rep(0, 4), obj, method="Nelder-Mead", control=list(maxit=150))
    rt_pred <- extract_baseline_exgauss_sim(res$par, resp, out, rt)
    
    w1 <- mean(abs(sort(rt_pred) - sort(rt)))
    beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
    
    list(w1 = w1, beta = beta_sim, rt_pred = rt_pred, trials = nrow(d), par=res$par)
})
all_base_w1 <- sum(unlist(lapply(base_eval, function(x) x$w1))) / S
all_base_beta <- mean(unlist(lapply(base_eval, function(x) x$beta)), na.rm=TRUE)

cat("Phase 2: Baseline Composite Rademacher Fit...\n")
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
    optim(rep(0, 4), obj, method="Nelder-Mead", control=list(maxit=150))$value
})

hyper_list <- list()
hyper_list[[1]] <- c(0.01, 1.0, 5.0)
hyper_list[[2]] <- c(0.01, 0.5, 5.0)
hyper_list[[3]] <- c(0.05, 1.0, 5.0)
hyper_list[[4]] <- c(0.1, 0.9, 5.0)
hyper_list[[5]] <- c(0.01, 1.0, 10.0)
hyper_list[[6]] <- c(0.01, 0.5, 10.0)
hyper_list[[7]] <- c(0.05, 1.0, 10.0)
hyper_list[[8]] <- c(0.1, 0.9, 10.0)
hyper_list[[9]] <- c(0.01, 1.0, 2.0)
hyper_list[[10]] <- c(0.1, 0.9, 2.0)

fileConn <- file("magi_ledger.md", "a")
writeLines("\n## Epoch: TERMINAL (Composite LFI Matrix) | 2026-08-25\n\n### 1. MAGI Consensus (The Final Pareto Convergence)\n*   **Caspar:** Aborting all singular objective losses. We must enforce simultaneous biological geometry and temporal fatigue mapping. By blending the 1-Wasserstein distribution geometry with the sequence beta coefficient, we construct the terminal composite loss.\n*   **Balthazar:** Gradient tracker shifted to Nelder-Mead (Derivative-Free Simplex). L-BFGS-B finite difference matrices will shatter on the stochastically sorted composite space. Nelder-Mead will map the final topological frontiers.\n\n### 2. Epoch Results Summary\n", fileConn)
close(fileConn)

rad_pass <- 0
dual_pass <- 0

for(var_idx in c(4, 9, 10)) {
    hyper <- hyper_list[[var_idx]]
    cat(sprintf("Evaluating Variant %d (L_min=%.2f, L_max=%.2f, Pois=%.1f)...\n", var_idx, hyper[1], hyper[2], hyper[3]))
    
    cand_eval <- lapply(1:S, function(s_idx) {
        d <- dat_clean %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
        obj <- function(phi) {
            rt_sim <- extract_epoch2_wasserstein(phi, hyper, resp, out, rt)
            w1 <- mean(abs(sort(rt_sim) - sort(rt)))
            beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
            if(is.na(beta_sim)) beta_sim <- 0
            return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
        }
        res <- optim(rep(0, 9), obj, method="Nelder-Mead", control=list(maxit=150))
        rt_pred <- extract_epoch2_wasserstein(res$par, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_pred) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
        list(w1 = w1, beta = beta_sim, rt_pred = rt_pred, trials = nrow(d), par=res$par, obj_val = res$value)
    })
    
    total_cand_w1 <- sum(unlist(lapply(cand_eval, function(x) x$w1))) / S
    total_cand_beta <- mean(unlist(lapply(cand_eval, function(x) x$beta)), na.rm=TRUE)
    
    cand_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
        d <- dat_noise %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
        obj <- function(phi) {
            rt_sim <- extract_epoch2_wasserstein(phi, hyper, resp, out, rt)
            w1 <- mean(abs(sort(rt_sim) - sort(rt)))
            beta_sim <- suppressWarnings(coef(lm(rt ~ rt_sim))["rt_sim"])
            if(is.na(beta_sim)) beta_sim <- 0
            return(lambda_1 * w1 + lambda_2 * abs(0.5 - beta_sim) + lambda_reg * sum(phi^2))
        }
        optim(rep(0, 9), obj, method="Nelder-Mead", control=list(maxit=150))$value
    })
    
    rad_diff <- unlist(cand_noise_eval) - unlist(base_noise_eval)
    t_test_rad <- t.test(rad_diff, alternative="less") 
    rad_p <- t_test_rad$p.value 
    
    cat(sprintf("Var %d: Rad_p=%.3f, W1_Cand=%.4f (Base=%.4f), Beta_Cand=%.4f (Base=%.4f)\n", 
                var_idx, rad_p, total_cand_w1, all_base_w1, total_cand_beta, all_base_beta))
}

fileConn <- file("magi_ledger.md", "a")
writeLines("### TERMINAL STATUS REACHED.\n", fileConn)
close(fileConn)
cat("Terminal Epoch Complete.\n")
