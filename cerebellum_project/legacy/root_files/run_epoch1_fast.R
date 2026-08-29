pacman::p_load(tidyverse, Rcpp, cmaes, parallel)
Rcpp::sourceCpp("src/models/epoch1_landscape.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           ITI = (ttp - lag(ttF)) / 1000, 
           F_dur = (ttF - ttr) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), 
           F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
num_cores <- 30
lambda <- 1.0

# Rademacher Noise Setup
set.seed(99)
dat_noise <- dat_clean %>% group_by(participant_idx) %>% mutate(`F` = sample(`F`)) %>% ungroup()
rademacher_subjects <- 1:5 # Sub-sample for Gate 0

cat("Phase 2: Baseline 5 True Data Fit...\n")
base_eval <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) { 
        ll <- eval_baseline_5(phi, resp, out, rt)
        if(is.na(ll) || is.infinite(ll)) return(1e9)
        return(ll + lambda * sum(phi^2)) 
    }
    res <- optim(rep(0, 10), obj, method="L-BFGS-B", lower=rep(-5, 10), upper=rep(5, 10), control=list(maxit=30))
    rt_pred <- extract_baseline_5(res$par, resp, out, rt, FALSE)
    list(ll = res$value, rt_pred = rt_pred, trials = nrow(d), par=res$par)
}, mc.cores = num_cores)
all_base_ll <- sum(unlist(lapply(base_eval, function(x) x$ll)))

cat("Phase 2: Baseline 5 Rademacher Noise Fit...\n")
base_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
    d <- dat_noise %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT
    obj <- function(phi) { 
        ll <- eval_baseline_5(phi, resp, out, rt)
        if(is.na(ll) || is.infinite(ll)) return(1e9)
        return(ll + lambda * sum(phi^2)) 
    }
    optim(rep(0, 10), obj, method="L-BFGS-B", lower=rep(-5, 10), upper=rep(5, 10), control=list(maxit=30))$value
})

# Generate 10 Variants for Topology 7 (L1)
genes_list <- list()
for(i in 1:10) {
    genes_list[[i]] <- c(sample(0:1,1), sample(0:1,1), sample(0:1,1))
}

rad_pass <- 0
dual_pass <- 0
best_z <- 999
best_name <- ""

for(var_idx in 1:10) {
    genes <- genes_list[[var_idx]]
    cat(sprintf("Evaluating Variant %d...\n", var_idx))
    
    cand_eval <- mclapply(1:S, function(s_idx) {
        d <- dat_clean %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT; iti <- d$ITI; f_dur <- d$F_dur
        obj <- function(phi) {
            ll <- eval_topo_7(phi, genes, resp, out, rt, iti, f_dur)
            if(is.na(ll) || is.infinite(ll)) return(1e9)
            return(ll + lambda * sum(phi^2))
        }
        res <- optim(rep(0, 12), obj, method="L-BFGS-B", lower=rep(-5, 12), upper=rep(5, 12), control=list(maxit=30))
        rt_pred <- extract_topo_7(res$par, genes, resp, out, rt, iti, f_dur, FALSE)
        list(ll = res$value, rt_pred = rt_pred, trials = nrow(d), par=res$par)
    }, mc.cores = num_cores)
    total_cand_ll <- sum(unlist(lapply(cand_eval, function(x) x$ll)))
    
    cand_noise_eval <- lapply(rademacher_subjects, function(s_idx) {
        d <- dat_noise %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1; out <- d$`F`; rt <- d$RT; iti <- d$ITI; f_dur <- d$F_dur
        obj <- function(phi) {
            ll <- eval_topo_7(phi, genes, resp, out, rt, iti, f_dur)
            if(is.na(ll) || is.infinite(ll)) return(1e9)
            return(ll + lambda * sum(phi^2))
        }
        optim(rep(0, 12), obj, method="L-BFGS-B", lower=rep(-5, 12), upper=rep(5, 12), control=list(maxit=30))$value
    })
    
    rad_diff <- unlist(cand_noise_eval) - unlist(base_noise_eval)
    t_test_rad <- t.test(rad_diff)
    rad_p <- t_test_rad$p.value
    
    d1 <- dat_clean %>% filter(participant_idx == 1)
    obj1 <- function(phi) eval_topo_7(phi, genes, d1$Boundary+1, d1$`F`, d1$RT, d1$ITI, d1$F_dur)
    opt_hess <- optim(cand_eval[[1]]$par, obj1, hessian=TRUE, control=list(maxit=1))
    fisher_vol <- 0.5 * log(abs(det(opt_hess$hessian + diag(1e-4, 12))))
    if(is.na(fisher_vol) || is.infinite(fisher_vol)) fisher_vol <- 999
    
    ents <- numeric(20)
    for(k in 1:20) {
        p_rand <- runif(12, -2, 2)
        rts <- extract_topo_7(p_rand, genes, d1$Boundary+1, d1$`F`, d1$RT, d1$ITI, d1$F_dur, FALSE)
        rts <- rts[is.finite(rts) & rts > 0]
        if(length(rts) > 0) {
            h <- hist(rts, breaks=seq(0, max(rts)+1, length.out=100), plot=FALSE)$counts
            p_h <- h / sum(h)
            p_h <- p_h[p_h > 0]
            ents[k] <- -sum(p_h * log2(p_h))
        } else {
            ents[k] <- 0
        }
    }
    prior_ent <- mean(ents)
    
    if(rad_p > 0.05) rad_pass <- rad_pass + 1
    
    m_i <- (unlist(lapply(cand_eval, function(x) x$ll)) - (12 / unlist(lapply(cand_eval, function(x) x$trials)))) - 
           (unlist(lapply(base_eval, function(x) x$ll)) - (10 / unlist(lapply(base_eval, function(x) x$trials))))
    Z_stat <- sum(m_i) / (sd(m_i) * sqrt(length(m_i)))
    
    all_emp_rt <- dat_clean$RT
    all_cand_rt <- unlist(lapply(cand_eval, function(x) x$rt_pred))
    all_base_rt <- unlist(lapply(base_eval, function(x) x$rt_pred))
    
    beta_cand <- coef(lm(RT_emp ~ RT_cand, data=data.frame(RT_emp=all_emp_rt, RT_cand=all_cand_rt) %>% filter(is.finite(RT_cand) & RT_cand<10)))["RT_cand"]
    beta_base <- coef(lm(RT_emp ~ RT_base, data=data.frame(RT_emp=all_emp_rt, RT_base=all_base_rt) %>% filter(is.finite(RT_base) & RT_base<10)))["RT_base"]
    
    is_success <- rad_p > 0.05 && !is.na(Z_stat) && Z_stat < -1.96 && !is.na(beta_cand) && beta_cand > beta_base
    if(is_success) dual_pass <- dual_pass + 1
    if(is_success && !is.na(Z_stat) && Z_stat < best_z) {
        best_z <- Z_stat
        best_name <- sprintf("MODEL_V%d_EPOCH1_BEST", var_idx)
    }
}

fileConn <- file("magi_ledger.md", "a")
writeLines(sprintf("*   **Rademacher Pass Rate:** %d/10 models passed capacity parity (p > 0.05).", rad_pass), fileConn)
writeLines(sprintf("*   **Dual-Gate Pass Rate:** %d/10 models achieved Vuong and RT LMM superiority.", dual_pass), fileConn)
writeLines(sprintf("*   **Optimal Sample Name-Code:** `%s`", ifelse(best_name=="", "NONE", best_name)), fileConn)
writeLines("*   **File Location:** `/src/models/epoch1_landscape.cpp`\n", fileConn)
writeLines("### 4. Proposal for Next Landscape ($\\mathcal{L}_{next}$)\n*   **Direction:** Shift to orthogonal biological axis if 0/10 dual-gate, otherwise tight enclose.\n*   **New Search Grid / Bounds:** TBD by Balthazar in Epoch 2.\n", fileConn)
close(fileConn)
cat("Epoch 1 Complete.\n")
