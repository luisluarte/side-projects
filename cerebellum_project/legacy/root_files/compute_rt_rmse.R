pacman::p_load(tidyverse, Rcpp, cmaes, parallel, Metrics)
CORES <- parallel::detectCores()

Rcpp::sourceCpp("magi_thermo_sudoku_core.cpp")
Rcpp::sourceCpp("magi_ext.cpp")
Rcpp::sourceCpp("magi_grand_phylogeny.cpp") 

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=F)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0)) %>% 
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

h <- c(1, 0.8, 20) 

run_subject_rmse <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        
        obj_base <- function(p) { v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_base <- tryCatch(cma_es(rep(0,4), obj_base, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,4), value=NA))
        ext_b <- ext_base(res_base$par, d$Boundary+1, d$Reward, d$RT)
        rmse_base <- Metrics::rmse(d$RT, ext_b[,1])
        
        obj_cand <- function(p) { v <- get_nll_thermo_sudoku(p, h, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_cand <- tryCatch(cma_es(rep(0,7), obj_cand, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,7), value=NA))
        ext_c <- ext_thermo_sudoku(res_cand$par, h, d$Boundary+1, d$Reward, d$RT)
        rmse_cand <- Metrics::rmse(d$RT, ext_c[,1])
        
        data.frame(SubjectID=s_idx, RMSE_Base=rmse_base, RMSE_Cand=rmse_cand)
    }, error = function(e) data.frame(SubjectID=s_idx, RMSE_Base=NA, RMSE_Cand=NA))
}

cat("Calculating RMSE for CC_Model_069 vs Baseline...\n")
df_rmse <- bind_rows(mclapply(1:S, run_subject_rmse, mc.cores=CORES)) %>% drop_na()

mean_base <- mean(df_rmse$RMSE_Base)
mean_cand <- mean(df_rmse$RMSE_Cand)
wt <- wilcox.test(df_rmse$RMSE_Cand, df_rmse$RMSE_Base, paired=TRUE)

cat("======================================\n")
cat("CC_Model_069 vs Baseline RT RMSE\n")
cat("======================================\n")
cat(sprintf("Marginal Mean RMSE (Baseline) : %.4f seconds\n", mean_base))
cat(sprintf("Marginal Mean RMSE (CC_069)   : %.4f seconds\n", mean_cand))
cat(sprintf("Mean Difference (Cand - Base) : %.4f seconds\n", mean_cand - mean_base))
cat(sprintf("Wilcoxon Paired p-value       : %.2e\n", wt$p.value))
cat("======================================\n")
