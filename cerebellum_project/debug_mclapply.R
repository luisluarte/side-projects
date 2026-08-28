pacman::p_load(tidyverse, Rcpp, cmaes, parallel)
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
d_list <- split(dat_clean, dat_clean$participant_idx)

run_base <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p) { v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res <- tryCatch(cma_es(rep(0,4), obj, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,4), value=NA))
        data.frame(SubjectID=s_idx, Base_NLL=res$value)
    }, error = function(e) as.character(e))
}

res <- mclapply(1:10, run_base, mc.cores=CORES)
print(res)
