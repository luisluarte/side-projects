pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick)

CORES <- parallel::detectCores()
Rcpp::sourceCpp("magi_thermo_sudoku_core.cpp")
Rcpp::sourceCpp("magi_nll_sweep.cpp") # for Baseline DDM

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    dplyr::mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% dplyr::filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    dplyr::mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

df_grid <- expand.grid(
    struct_id = c(1, 2, 3, 4), 
    kappa = c(0.1, 0.5, 0.8, 0.9, 0.99), 
    K_sa = c(2, 5, 10, 20, 50) 
)
df_grid$model_idx <- 1:100
df_grid$model_id <- sprintf("CC_Model_%03d", df_grid$model_idx)

cat("Starting Thermodynamic Sudoku Framework Sweep (100 Models)...\n")
out_file <- "results/tables/thermo_sudoku_sweep.csv"
if(!file.exists(out_file)) write_csv(data.frame(SubjectID=integer(), ModelID=character(), NLL=numeric()), out_file)

run_subj <- function(s_idx, h) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p) {
            v <- get_nll_thermo_sudoku(p, h, d$Boundary+1, d$Reward, d$RT)
            if(is.nan(v)) return(1e6) else return(v)
        }
        res <- tryCatch(cma_es(rep(0, 7), obj, control=list(maxit=50, sigma=0.5)), error = function(e) list(value=NA))
        return(res$value)
    }, error = function(e) return(NA))
}

for(i in 1:nrow(df_grid)) {
    mod_id <- df_grid$model_id[i]
    h <- c(df_grid$struct_id[i], df_grid$kappa[i], df_grid$K_sa[i])
    
    cat("Evaluating", mod_id, "\n")
    nll_vals <- mclapply(1:S, run_subj, h=h, mc.cores=CORES)
    
    df_res <- data.frame(SubjectID=1:S, ModelID=mod_id, NLL=unlist(nll_vals))
    write_csv(df_res, out_file, append=TRUE)
}
cat("SWEEP INITIATED OR COMPLETED.\n")
