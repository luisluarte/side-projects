pacman::p_load(tidyverse, Rcpp, cmaes, parallel, lmerTest)
CORES <- parallel::detectCores()
Rcpp::sourceCpp("magi_swarm_005.cpp")
Rcpp::sourceCpp("magi_swarm_001.cpp") # baseline

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=F)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0)) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

cat("Phase 1: Evaluating Baseline WFPT...\n")
run_base <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p){v<-get_nll_base_w(p,d$Boundary+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res <- tryCatch(cma_es(rep(0,4),obj,control=list(maxit=50,sigma=0.5)),error=function(e)list(par=rep(0,4),value=NA))
        data.frame(SubjectID=s_idx, NLL=res$value, Model="Base")
    }, error=function(e) data.frame(SubjectID=s_idx, NLL=NA, Model="Base"))
}
df_base <- bind_rows(mclapply(1:S, run_base, mc.cores=CORES))

cat("Phase 2: Evaluating Swarm 005...\n")
run_cand <- function(s_idx, h) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p){v<-get_nll_swarm_005(p,h,d$Boundary+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res <- tryCatch(cma_es(rep(0,7),obj,control=list(maxit=50,sigma=0.5)),error=function(e)list(par=rep(0,7),value=NA))
        data.frame(SubjectID=s_idx, NLL=res$value, Model="Swarm005")
    }, error=function(e) data.frame(SubjectID=s_idx, NLL=NA, Model="Swarm005"))
}

# h_swarm <- c(lambda_sparse, beta_ising, K_sa)
h_swarm <- c(4.64e-4, 5e-4, 18)
df_cand <- bind_rows(mclapply(1:S, run_cand, h=h_swarm, mc.cores=CORES))

df_all <- bind_rows(df_base, df_cand) %>% drop_na()
df_all$Model <- factor(df_all$Model, levels=c("Base", "Swarm005"))

cat("\n========== STATISTICAL SUPREMACY TEST ==========\n")
if(nrow(df_all) > 10) {
    mod <- lmer(NLL ~ Model + (1 | SubjectID), data=df_all)
    sm <- summary(mod)
    est <- sm$coefficients["ModelSwarm005", "Estimate"]
    t_val <- sm$coefficients["ModelSwarm005", "t value"]
    p_val <- sm$coefficients["ModelSwarm005", "Pr(>|t|)"]
    
    N_valid <- length(unique(df_all$SubjectID))
    d <- -1.0 * (t_val / sqrt(N_valid)) # Negative because lower NLL is better
    
    cat(sprintf("LMM Estimate (Delta NLL): %.3f\n", est))
    cat(sprintf("LMM t-value: %.3f\n", t_val))
    cat(sprintf("LMM p-value: %.2e\n", p_val))
    cat(sprintf("Cohen's d: %.3f\n\n", d))
    
    if(d >= 0.5 && p_val < 0.05) {
        cat("SUPREMACY ACHIEVED (d >= 0.5).\n")
        write_lines("SUPREMACY ACHIEVED", "swarm_supremacy_flag.txt")
    } else {
        cat("SUPREMACY FAILED. Diagnostic protocol required.\n")
    }
} else {
    cat("Insufficient data for LMM.\n")
}
