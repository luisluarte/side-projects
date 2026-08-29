pacman::p_load(tidyverse, Rcpp, cmaes)
Rcpp::sourceCpp("src/models/epoch2_wasserstein_landscape.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- 30
dat_clean <- dat_clean %>% filter(participant_idx <= S)

lambda_1 <- 1.0
lambda_2 <- 5.0
lambda_reg <- 0.05

hyper_list <- list()
hyper_list[[4]] <- c(0.10, 0.90, 5.0) # High noise (MLE Beta Champion)
hyper_list[[9]] <- c(0.01, 1.00, 2.0) # Low noise

cat("Evaluating Uncoupled Ex-Gaussian Surrogate (LFI) with CMA-ES (N=30)...\n")
cat("Reference Baseline (N=30): W1 = 0.2418, Beta = 0.2723\n")

for(var_idx in c(4, 9)) {
    hyper <- hyper_list[[var_idx]]
    cat(sprintf("\n--- Evaluating Variant %d (L_min=%.2f, L_max=%.2f, Pois=%.1f) ---\n", var_idx, hyper[1], hyper[2], hyper[3]))
    
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
        
        res <- cma_es(rep(0, 9), obj, lower=rep(-5, 9), upper=rep(5, 9), control=list(maxit=150))
        
        rt_pred <- extract_epoch2_wasserstein(res$par, hyper, resp, out, rt)
        w1 <- mean(abs(sort(rt_pred) - sort(rt)))
        beta_sim <- suppressWarnings(coef(lm(rt ~ rt_pred))["rt_pred"])
        
        list(w1 = w1, beta = beta_sim)
    })
    
    total_cand_w1 <- sum(unlist(lapply(cand_eval, function(x) x$w1))) / S
    total_cand_beta <- mean(unlist(lapply(cand_eval, function(x) x$beta)), na.rm=TRUE)
    
    cat(sprintf(">>> Variant %d (CMA-ES): W1_Cand=%.4f, Beta_Cand=%.4f\n", var_idx, total_cand_w1, total_cand_beta))
}
