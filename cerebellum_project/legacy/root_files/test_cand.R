pacman::p_load(tidyverse, Rcpp, cmaes, yardstick)
Rcpp::sourceCpp("magi_thermo_sudoku_core.cpp")
Rcpp::sourceCpp("magi_ext.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=F)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0)) %>% 
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id)))
d_list <- split(dat_clean, dat_clean$participant_idx)

s_idx <- 1
d <- d_list[[s_idx]]
h <- c(1, 0.8, 20)

tryCatch({
    obj <- function(p) { v <- get_nll_thermo_sudoku(p, h, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
    res <- tryCatch(cma_es(rep(0, 7), obj, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,7), value=NA))
    print("CMA-ES 1 DONE")
    mets <- c(PR_AUC=NA, ROC_AUC=NA, MCC=NA, Brier=NA)
    if(!is.na(res$value)) {
        ext <- ext_thermo_sudoku(res$par, h, d$Boundary+1, d$Reward, d$RT)
        print("EXT 1 DONE")
        # calc_mets is not defined here, let's just mock it
    }
    set.seed(s_idx)
    R_noise <- sample(c(0, 1), nrow(d), replace=TRUE)
    obj_rad <- function(p) { v <- get_nll_thermo_sudoku(p, h, R_noise+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
    res_rad <- tryCatch(cma_es(rep(0, 7), obj_rad, control=list(maxit=30, sigma=0.5)), error=function(e) list(par=rep(0,7)))
    print("CMA-ES RAD DONE")
    ext_rad <- ext_thermo_sudoku(res_rad$par, h, R_noise+1, d$Reward, d$RT)
    print("EXT RAD DONE")
}, error = function(e) print(e))
