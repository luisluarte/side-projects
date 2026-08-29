pacman::p_load(tidyverse, Rcpp, cmaes, parallel, lmerTest)
CORES <- parallel::detectCores()
Rcpp::sourceCpp("magi_swarm_006.cpp")

# Note: The data pipeline must compute ITI (Inter-Trial Interval)
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=F)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(
        RT = (ttr-ttp)/1000, 
        Boundary = ifelse(Resp==2, 1, 0),
        ITI = (ttp - lag(ttr))/1000 # Compute physical time gap
    ) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(
        ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI),
        participant_idx = as.integer(as.factor(participant_id))
    )
S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

cat("Phase 1: Evaluating Swarm 006 (Continuous Physical ITI + Epistemic Boundary)...\n")
run_cand <- function(s_idx, h) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p){v<-get_nll_swarm_006(p,h,d$Boundary+1,d$Reward,d$RT,d$ITI);if(is.nan(v))1e6 else v}
        res <- tryCatch(cma_es(rep(0,9),obj,control=list(maxit=50,sigma=0.5)),error=function(e)list(par=rep(0,9),value=NA))
        data.frame(SubjectID=s_idx, NLL=res$value, Model="Swarm006")
    }, error=function(e) data.frame(SubjectID=s_idx, NLL=NA, Model="Swarm006"))
}

# h_swarm <- c(lambda_sparse, beta_ising, K_sa)
h_swarm <- c(4.64e-4, 5e-4, 18)
df_cand <- bind_rows(mclapply(1:S, run_cand, h=h_swarm, mc.cores=CORES))

cat("Optimization complete.\n")
print(head(df_cand))
