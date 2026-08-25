pacman::p_load(tidyverse, Rcpp, optimx, parallel)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/extract_ql_fatigue.cpp")
Rcpp::sourceCpp("src/models/eccm_magi_topologies.cpp")

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

cat("Pre-computing Baseline (Dynamic QL Poly Fatigue)...\n")
baseline_eval <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    obj_ql <- function(phi) { return(eval_ql_ddm_dynamic_poly_fatigue(phi, resp, out, rt) + lambda * sum(phi^2)) }
    res <- optim(rep(0, 9), obj_ql, method="L-BFGS-B", lower=rep(-5, 9), upper=rep(5, 9))
    ll <- extract_ll_ql_dynamic_poly_fatigue(res$par, resp, out, rt)
    rt_pred <- extract_rt_ql_dynamic_poly_fatigue(res$par, resp, out, rt)
    return(list(ll = ll, rt_pred = rt_pred, emp_rt = rt, pid = d$participant_id, trials = nrow(d)))
}, mc.cores = num_cores)

all_baseline_ll <- unlist(lapply(baseline_eval, function(x) x$ll))
all_emp_rt <- unlist(lapply(baseline_eval, function(x) x$emp_rt))
all_base_rt <- unlist(lapply(baseline_eval, function(x) x$rt_pred))
all_pid <- unlist(lapply(baseline_eval, function(x) rep(x$pid[1], x$trials)))
T_total <- length(all_baseline_ll)

# Start clean log
fileConn <- file("live_audit_log.md", "w")
writeLines("# MAGI Meta-Optimization Ledger\n", fileConn)
close(fileConn)

set.seed(1337)
topo_names <- c("Independent Dual-Kernel", "Cerebellar Cortical-Gating", "Competitive Dual-Kernel")

for(topo in 1:3) {
    cat(sprintf("\n>>> STARTING 100 ITERATIONS FOR TOPOLOGY %d: %s <<<\n", topo, topo_names[topo]))
    
    genes_list <- list()
    for(i in 1:100) {
        genes_list[[i]] <- c(
            topo,
            sample(0:1, 1), # G1 Conflict Type
            sample(0:1, 1), # G2 Drift Arousal
            sample(0:2, 1), # G3 Boundary Mech
            sample(0:1, 1), # G4 CB Pearce-Hall
            sample(0:1, 1), # G5 T_nd Mech
            sample(c(5, 10, 20, 40), 1) # N_MF
        )
    }

    magi_results <- mclapply(1:100, function(iter) {
        genes <- genes_list[[iter]]
        
        mask <- rep(1, 16)
        if(genes[3] == 0) mask[13] <- 0
        if(genes[4] == 0) mask[9] <- 0
        if(genes[4] == 1) mask[12] <- 0
        if(genes[4] == 2) mask[16] <- 0
        if(genes[5] == 0) mask[14] <- 0
        if(genes[6] == 0) mask[15] <- 0
        
        iter_ll <- numeric(T_total)
        iter_rt <- numeric(T_total)
        idx_start <- 1
        
        for(s_idx in 1:S) {
            d <- dat_clean %>% filter(participant_idx == s_idx)
            resp <- d$Boundary + 1
            out <- d$`F`
            rt <- d$RT
            iti <- d$ITI
            f_dur <- d$F_dur
            ttp <- d$ttp
            T_trials <- nrow(d)
            
            obj_fn <- function(phi) {
                phi_masked <- phi * mask
                return(eval_magi_topo(phi_masked, genes, resp, out, rt, iti, f_dur, ttp) + lambda * sum(phi_masked^2))
            }
            
            res <- optim(rep(0, 16), obj_fn, method="L-BFGS-B", lower=rep(-5, 16), upper=rep(5, 16))
            phi_opt <- res$par * mask
            
            ll <- extract_topology(phi_opt, genes, resp, out, rt, iti, f_dur, ttp, TRUE)
            rt_pred <- extract_topology(phi_opt, genes, resp, out, rt, iti, f_dur, ttp, FALSE)
            
            idx_end <- idx_start + T_trials - 1
            iter_ll[idx_start:idx_end] <- ll
            iter_rt[idx_start:idx_end] <- rt_pred
            idx_start <- idx_end + 1
        }
        
        m_i <- (iter_ll - (16/T_trials)) - (all_baseline_ll - (9/T_trials)) 
        m_i <- m_i[is.finite(m_i)]
        Z_stat <- sum(m_i) / (sd(m_i) * sqrt(length(m_i)))
        p_z <- 2 * (1 - pnorm(abs(Z_stat)))
        
        df_lmm <- data.frame(RT_emp = all_emp_rt, RT_cand = iter_rt, RT_base = all_base_rt, PID = all_pid)
        df_lmm <- df_lmm %>% filter(is.finite(RT_cand) & RT_cand > 0 & RT_cand < 10)
        
        lm_base <- lm(RT_emp ~ RT_base, data=df_lmm)
        lm_cand <- lm(RT_emp ~ RT_cand, data=df_lmm)
        beta_base <- coef(lm_base)["RT_base"]
        beta_cand <- coef(lm_cand)["RT_cand"]
        
        md_text <- sprintf("## Topology %d | Iteration: %d | 2026-08-25T03:35:00-04:00\n\n", topo, iter)
        md_text <- paste0(md_text, "### 1. MAGI Consensus\n")
        md_text <- paste0(md_text, "*   **Melchior:** Permutation: G1=", genes[2], ", G2=", genes[3], ", G3=", genes[4], ", G4=", genes[5], ", G5=", genes[6], ".\n")
        md_text <- paste0(md_text, "*   **Caspar:** N_MF=", genes[7], ".\n")
        md_text <- paste0(md_text, "*   **Balthazar (Synthesis):** Executing automated combinatorial grid point.\n\n")
        md_text <- paste0(md_text, "### 2. The Proposal (Mutation Landscape)\n")
        md_text <- paste0(md_text, "*   **Mathematical Formulation:** [Variant ", paste(genes, collapse="-"), "]\n")
        md_text <- paste0(md_text, "*   **Search Grid / Bounds:** L-BFGS-B constrained manifold.\n\n")
        md_text <- paste0(md_text, "### 3. Name-Code\n*   `MODEL_TOPO_", topo, "_VAR_", iter, "`\n\n")
        md_text <- paste0(md_text, "### 4. File Location\n*   `/src/models/eccm_magi_topologies.cpp`\n\n")
        md_text <- paste0(md_text, "### 5. Results & Dual-Gate Metrics\n")
        md_text <- paste0(md_text, "*   **Vuong Z-Statistic:** ", round(Z_stat, 3), " (p = ", signif(p_z, 3), ")\n")
        md_text <- paste0(md_text, "*   **LMM Beta Contrast:** Cand=", round(beta_cand, 4), " vs Base=", round(beta_base, 4), "\n")
        
        is_success <- is.finite(Z_stat) && Z_stat > 1.96 && beta_cand > beta_base
        success <- ifelse(is_success, "Success: Enclosing landscape. CHAMPION FOUND!", "Failure: Expanding landscape.")
        md_text <- paste0(md_text, "*   **MAGI Feedback:** ", success, "\n\n---\n")
        
        return(list(text = md_text, success = is_success))
    }, mc.cores = num_cores)

    fileConn <- file("live_audit_log.md", "a")
    champion_found <- FALSE
    for(res in magi_results) {
        writeLines(res$text, fileConn)
        if (res$success) {
            champion_found <- TRUE
        }
    }
    close(fileConn)
    
    if (champion_found) {
        cat(sprintf("\n>>> CHAMPION DISCOVERED IN TOPOLOGY %d! HALTING MAGI SEARCH <<<\n", topo))
        break
    } else {
        cat(sprintf("\n>>> TOPOLOGY %d FAILED AFTER 100 ITERATIONS. MOVING TO NEXT TOPOLOGY <<<\n", topo))
    }
}
cat("MAGI Search Complete.\n")
