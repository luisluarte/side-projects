pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick)

CORES <- parallel::detectCores()
Rcpp::sourceCpp("magi_thermo_sudoku_core.cpp")
Rcpp::sourceCpp("magi_ext.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    dplyr::mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% dplyr::filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    dplyr::mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

df_grid <- expand.grid(struct_id=c(1,2,3,4), kappa=c(0.1,0.5,0.8,0.9,0.99), K_sa=c(2,5,10,20,50))
df_grid$model_idx <- 1:100
df_grid$model_id <- sprintf("CC_Model_%03d", df_grid$model_idx)

out_file <- "results/tables/thermo_final_metrics.csv"
if(!file.exists(out_file)) {
    write_csv(data.frame(SubjectID=integer(), ModelID=character(), NLL=numeric(), PR_AUC=numeric(), ROC_AUC=numeric(), MCC=numeric(), Brier=numeric(), Rademacher=numeric()), out_file)
}

calc_mets <- function(probs, y_true) {
    tryCatch({
        df <- data.frame(truth=factor(y_true, levels=c("1", "0")), p1=probs)
        df$pred_class <- factor(ifelse(df$p1 > 0.5, "1", "0"), levels=c("1", "0"))
        pr <- pr_auc(df, truth, p1)$.estimate
        roc <- roc_auc(df, truth, p1)$.estimate
        m <- mcc(df, truth, pred_class)$.estimate
        brier <- brier_class(df, truth, p1)$.estimate
        c(PR_AUC=pr, ROC_AUC=roc, MCC=m, Brier=brier)
    }, error=function(e) c(PR_AUC=NA, ROC_AUC=NA, MCC=NA, Brier=NA))
}

run_subj <- function(s_idx, h) {
    tryCatch({
        d <- d_list[[s_idx]]
        y_true <- d$Boundary
        
        obj <- function(p) { v <- get_nll_thermo_sudoku(p, h, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res <- tryCatch(cma_es(rep(0, 7), obj, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,7), value=NA))
        
        mets <- c(PR_AUC=NA, ROC_AUC=NA, MCC=NA, Brier=NA)
        if(!is.na(res$value)) {
            ext <- ext_thermo_sudoku(res$par, h, d$Boundary+1, d$Reward, d$RT)
            mets <- calc_mets(ext[,1], y_true)
        }
        
        # Rademacher: fit to random choice vector
        set.seed(s_idx)
        R_noise <- sample(c(0, 1), nrow(d), replace=TRUE)
        obj_rad <- function(p) { v <- get_nll_thermo_sudoku(p, h, R_noise+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_rad <- tryCatch(cma_es(rep(0, 7), obj_rad, control=list(maxit=30, sigma=0.5)), error=function(e) list(par=rep(0,7)))
        ext_rad <- ext_thermo_sudoku(res_rad$par, h, R_noise+1, d$Reward, d$RT)
        rad_cor <- abs(cor(ext_rad[,1], R_noise))
        if(is.na(rad_cor)) rad_cor <- 0
        
        return(c(SubjectID=s_idx, NLL=res$value, mets, Rademacher=rad_cor))
    }, error = function(e) return(c(SubjectID=s_idx, NLL=NA, PR_AUC=NA, ROC_AUC=NA, MCC=NA, Brier=NA, Rademacher=NA)))
}

cat("Extracting exact parameter topologies and Rademacher limits...\n")
for(i in 1:nrow(df_grid)) {
    mod_id <- df_grid$model_id[i]
    h <- c(df_grid$struct_id[i], df_grid$kappa[i], df_grid$K_sa[i])
    
    cat("Evaluating", mod_id, "\n")
    res_list <- mclapply(1:S, run_subj, h=h, mc.cores=CORES)
    
    df_res <- as.data.frame(do.call(rbind, res_list))
    df_res$ModelID <- mod_id
    df_res <- df_res[, c("SubjectID", "ModelID", "NLL", "PR_AUC", "ROC_AUC", "MCC", "Brier", "Rademacher")]
    
    write_csv(df_res, out_file, append=TRUE)
}
cat("EXTRACTION COMPLETE.\n")
