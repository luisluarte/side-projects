pacman::p_load(tidyverse, Rcpp, cmaes, parallel)

Rcpp::sourceCpp("src/models/matched_baselines.cpp")
Rcpp::sourceCpp("src/models/eccm_magi_v5_v6.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           ITI = (ttp - lag(ttF)) / 1000, 
           F_dur = (ttF - ttr) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), 
           F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% 
    ungroup() %>% 
    filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 
num_cores <- 30
cmaes_maxit <- 50

cat("Pre-computing Baseline 5 (Reward Integrator, 10-param)...\n")
baseline5_eval <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    obj_base <- function(phi) { 
        ll <- eval_baseline_5(phi, resp, out, rt)
        if(is.na(ll) || is.infinite(ll)) return(1e9)
        return(ll + lambda * sum(phi^2)) 
    }
    res <- cma_es(rep(0, 10), obj_base, lower=rep(-5, 10), upper=rep(5, 10), control=list(maxit=cmaes_maxit))
    rt_pred <- extract_baseline_5(res$par, resp, out, rt, FALSE)
    return(list(ll = res$value, rt_pred = rt_pred, trials = nrow(d)))
}, mc.cores = num_cores)
all_base5_ll <- sum(unlist(lapply(baseline5_eval, function(x) x$ll)))

cat("Pre-computing Baseline 6 (Value Normalization, 8-param)...\n")
baseline6_eval <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    obj_base <- function(phi) { 
        ll <- eval_baseline_6(phi, resp, out, rt)
        if(is.na(ll) || is.infinite(ll)) return(1e9)
        return(ll + lambda * sum(phi^2)) 
    }
    res <- cma_es(rep(0, 8), obj_base, lower=rep(-5, 8), upper=rep(5, 8), control=list(maxit=cmaes_maxit))
    rt_pred <- extract_baseline_6(res$par, resp, out, rt, FALSE)
    return(list(ll = res$value, rt_pred = rt_pred, trials = nrow(d)))
}, mc.cores = num_cores)
all_base6_ll <- sum(unlist(lapply(baseline6_eval, function(x) x$ll)))

all_emp_rt <- dat_clean$RT

set.seed(42)
genes_list <- list()
for(topo in 5:6) { 
    for(i in 1:10) {
        genes_list[[length(genes_list) + 1]] <- c(
            topo, sample(0:1, 1), sample(0:1, 1), 3, sample(0:1, 1), sample(0:1, 1), sample(c(5, 10, 20), 1)
        )
    }
}

fileConn <- file("live_audit_log_matched.md", "w")
writeLines("# MAGI Design-Matched Topologies Search Ledger\n", fileConn)
writeLines(sprintf("Baseline 5 (Abstract Integrator) NLL: %.1f\n", all_base5_ll), fileConn)
writeLines(sprintf("Baseline 6 (Abstract Normalizer) NLL: %.1f\n\n", all_base6_ll), fileConn)
close(fileConn)

cat("Running 20 Variants (Topologies 5 and 6)...\n")
for(var_idx in 1:20) {
    genes <- genes_list[[var_idx]]
    topo <- genes[1]
    
    mask <- rep(1, 19)
    if(genes[3] == 0) mask[13] <- 0
    if(genes[4] == 0) mask[9] <- 0
    if(genes[4] == 1) mask[12] <- 0
    if(genes[4] == 2) mask[16] <- 0
    if(genes[5] == 0) mask[14] <- 0
    if(genes[6] == 0) mask[15] <- 0
    mask[12] <- 1; mask[16] <- 1; mask[17] <- 1; mask[18] <- 1; mask[19] <- 0
    
    cat(sprintf("Evaluating Variant %d/20 (Topology %d)...\n", var_idx, topo))
    
    cand_eval <- mclapply(1:S, function(s_idx) {
        d <- dat_clean %>% filter(participant_idx == s_idx)
        resp <- d$Boundary + 1
        out <- d$`F`
        rt <- d$RT
        iti <- d$ITI
        f_dur <- d$F_dur
        ttp <- d$ttp
        
        obj_cand <- function(phi) {
            p <- phi * mask
            ll <- eval_magi_topo_v5_v6(p, genes, resp, out, rt, iti, f_dur, ttp)
            if(is.na(ll) || is.infinite(ll)) return(1e9)
            return(ll + lambda * sum(p^2))
        }
        
        res <- cma_es(rep(0, 19), obj_cand, lower=rep(-5, 19), upper=rep(5, 19), control=list(maxit=cmaes_maxit))
        rt_pred <- extract_topology_v5_v6(res$par * mask, genes, resp, out, rt, iti, f_dur, ttp, FALSE)
        return(list(ll = res$value, rt_pred = rt_pred, trials = nrow(d)))
    }, mc.cores = num_cores)
    
    total_cand_ll <- sum(unlist(lapply(cand_eval, function(x) x$ll)))
    all_cand_rt <- unlist(lapply(cand_eval, function(x) x$rt_pred))
    
    df_lmm <- data.frame(RT_emp = all_emp_rt, RT_cand = all_cand_rt)
    df_lmm <- df_lmm %>% filter(is.finite(RT_cand) & RT_cand > 0 & RT_cand < 10)
    lm_cand <- lm(RT_emp ~ RT_cand, data=df_lmm)
    beta_cand <- coef(lm_cand)["RT_cand"]
    
    # Select appropriate baseline
    if(topo == 5) {
        baseline_eval <- baseline5_eval
        K_base <- 10
    } else {
        baseline_eval <- baseline6_eval
        K_base <- 8
    }
    K_cand <- 18
    
    # AIC penalty per subject average trial
    m_i <- (unlist(lapply(cand_eval, function(x) x$ll)) - (K_cand / unlist(lapply(cand_eval, function(x) x$trials)))) - 
           (unlist(lapply(baseline_eval, function(x) x$ll)) - (K_base / unlist(lapply(baseline_eval, function(x) x$trials))))
           
    Z_stat <- sum(m_i) / (sd(m_i) * sqrt(length(m_i)))
    p_z <- 2 * (1 - pnorm(abs(Z_stat)))
    
    all_base_rt <- unlist(lapply(baseline_eval, function(x) x$rt_pred))
    df_base <- data.frame(RT_emp = all_emp_rt, RT_base = all_base_rt)
    df_base <- df_base %>% filter(is.finite(RT_base) & RT_base > 0 & RT_base < 10)
    beta_base <- coef(lm(RT_emp ~ RT_base, data=df_base))["RT_base"]
    
    md_text <- sprintf("## Topology %d | Variant %d\n\n", topo, var_idx)
    md_text <- paste0(md_text, "### 1. The Proposal\n")
    md_text <- paste0(md_text, "*   **Mathematical Formulation:** [Variant ", paste(genes, collapse="-"), "]\n")
    md_text <- paste0(md_text, "### 2. Global Results vs Design-Matched Baseline\n")
    md_text <- paste0(md_text, "*   **Baseline NLL (Sum):** ", round(sum(unlist(lapply(baseline_eval, function(x) x$ll))), 1), "\n")
    md_text <- paste0(md_text, "*   **Candidate NLL (Sum):** ", round(total_cand_ll, 1), "\n")
    md_text <- paste0(md_text, "*   **Vuong Z-Statistic:** ", round(Z_stat, 3), " (p = ", signif(p_z, 3), ")\n")
    md_text <- paste0(md_text, "*   **Beta Contrast:** Cand=", round(beta_cand, 4), " vs Base=", round(beta_base, 4), "\n\n")
    
    is_success <- is.finite(Z_stat) && Z_stat < -1.96 && beta_cand > beta_base
    success <- ifelse(is_success, "Success: Biological Champion Found!", "Failure: Capacity Constraint Unmet.")
    md_text <- paste0(md_text, "*   **MAGI Feedback:** ", success, "\n\n---\n")
    
    fileConn <- file("live_audit_log_matched.md", "a")
    writeLines(md_text, fileConn)
    close(fileConn)
}
cat("Matched Baselines Search Complete.\n")
