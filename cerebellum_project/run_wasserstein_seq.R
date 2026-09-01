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
lambda <- 0.05 # Reduced L2 penalty for W_1 scale

set.seed(99)
dat_noise <- dat_clean %>% group_by(participant_idx) %>% mutate(`F` = sample(`F`)) %>% ungroup()
rademacher_subjects <- 1:5 

cat("Phase 2: Baseline Ex-Gauss Wasserstein Fit...\n")
base_eval <- lapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) { 
        w1 <- eval_baseline_wasserstein(phi, resp, out, rt)
        if(is.na(w1) || is.infinite(w1)) return(1e9)
        return(w1 + lambda * sum(phi^2)) 
    }
    res <- optim(rep(0, 4), obj, method="L-BFGS-B", lower=rep(-5, 4), upper=rep(5, 4), control=list(maxit=30))
    rt_pred <- extract_baseline_exgauss_sim(res$par, resp, out, rt)
    list(w1 = res$value, rt_pred = rt_pred, trials = nrow(d), par=res$par)
})
all_base_w1 <- sum(unlist(lapply(base_eval, function(x) x$w1))) / S
cat(sprintf("Baseline Mean W1: %.4f\n", all_base_w1))

cat("Phase 2: Baseline Ex-Gauss Rademacher Fit...\n")
base_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
    d <- dat_noise %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) { 
        w1 <- eval_baseline_wasserstein(phi, resp, out, rt)
        if(is.na(w1) || is.infinite(w1)) return(1e9)
        return(w1 + lambda * sum(phi^2)) 
    }
    optim(rep(0, 4), obj, method="L-BFGS-B", lower=rep(-5, 4), upper=rep(5, 4), control=list(maxit=30))$value
})

# Hyperparameters: Lambda_min, Lambda_max, Poisson_Rate
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
writeLines("\n## Epoch: 2 (Wasserstein Generative Matrix) | 2026-08-25\n\n### 1. MAGI Consensus (Distributional Alignment)\n*   **Caspar:** Aborting MLE. We optimize directly against the empirical inverse CDF via 1-Wasserstein Distance (Earth Mover). This forces the optimizer to match the global structure of the RT distribution without penalizing deterministic trial-by-trial errors.\n*   **Balthazar:** L2 (Wasserstein) defined. Base=Baseline Cortex ExGauss, Cand=MutantManifold ExGauss.\n\n### 2. Epoch Results Summary\n", fileConn)
close(fileConn)

rad_pass <- 0
dual_pass <- 0
best_w1 <- 999
best_name <- ""

for(var_idx in c(4, 9, 10)) {
    hyper <- hyper_list[[var_idx]]
    cat(sprintf("Evaluating Variant %d (L_min=%.2f, L_max=%.2f, Pois=%.1f)...\n", var_idx, hyper[1], hyper[2], hyper[3]))
    
    cand_eval <- lapply(1:S, function(s_idx) {
        d <- dat_clean %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
        obj <- function(phi) {
            w1 <- eval_epoch2_wasserstein(phi, hyper, resp, out, rt)
            if(is.na(w1) || is.infinite(w1)) return(1e9)
            return(w1 + lambda * sum(phi^2))
        }
        res <- optim(rep(0, 9), obj, method="L-BFGS-B", lower=rep(-5, 9), upper=rep(5, 9), control=list(maxit=30))
        rt_pred <- extract_epoch2_wasserstein(res$par, hyper, resp, out, rt)
        list(w1 = res$value, rt_pred = rt_pred, trials = nrow(d), par=res$par)
    })
    total_cand_w1 <- sum(unlist(lapply(cand_eval, function(x) x$w1))) / S
    
    cand_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
        d <- dat_noise %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
        obj <- function(phi) {
            w1 <- eval_epoch2_wasserstein(phi, hyper, resp, out, rt)
            if(is.na(w1) || is.infinite(w1)) return(1e9)
            return(w1 + lambda * sum(phi^2))
        }
        optim(rep(0, 9), obj, method="L-BFGS-B", lower=rep(-5, 9), upper=rep(5, 9), control=list(maxit=30))$value
    })
    
    # In W1, a *lower* value means better fit. 
    # To pass Rademacher, the difference in fit (W1_noise_cand - W1_noise_base) 
    # should NOT be significantly better (lower) for the candidate.
    # Alternatively, Rademacher parity means the candidate fits noise no better than baseline.
    rad_diff <- unlist(cand_noise_eval) - unlist(base_noise_eval)
    # A negative diff means candidate fits noise better (overfitting).
    t_test_rad <- t.test(rad_diff, alternative="less") 
    rad_p <- t_test_rad$p.value # p > 0.05 means we CANNOT conclude cand fits noise better
    
    all_emp_rt <- dat_clean$RT
    all_cand_rt <- unlist(lapply(cand_eval, function(x) x$rt_pred))
    all_base_rt <- unlist(lapply(base_eval, function(x) x$rt_pred))
    
    beta_cand <- coef(lm(RT_emp ~ RT_cand, data=data.frame(RT_emp=all_emp_rt, RT_cand=all_cand_rt) %>% filter(is.finite(RT_cand) & RT_cand<10)))["RT_cand"]
    beta_base <- coef(lm(RT_emp ~ RT_base, data=data.frame(RT_emp=all_emp_rt, RT_base=all_base_rt) %>% filter(is.finite(RT_base) & RT_base<10)))["RT_base"]
    
    is_success <- rad_p > 0.05 && !is.na(total_cand_w1) && total_cand_w1 < all_base_w1 && !is.na(beta_cand) && beta_cand > beta_base
    if(is_success) dual_pass <- dual_pass + 1
    if(is_success && !is.na(total_cand_w1) && total_cand_w1 < best_w1) {
        best_w1 <- total_cand_w1
        best_name <- sprintf("MODEL_V%d_W1_EXGAUSS", var_idx)
    }
    
    cat(sprintf("Var %d: Rad_p=%.3f, W1_Cand=%.4f, W1_Base=%.4f, Beta=%.4f (BaseBeta=%.4f)\n", var_idx, rad_p, total_cand_w1, all_base_w1, beta_cand, beta_base))
}

fileConn <- file("magi_ledger.md", "a")
writeLines(sprintf("*   **Optimal Sample Name-Code:** `%s`", ifelse(best_name=="", "NONE", best_name)), fileConn)
close(fileConn)
cat("Wasserstein Generation Complete.\n")
