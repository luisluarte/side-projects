pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick)
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

# WE ONLY DO TOP 10 MODELS TO SAVE TIME!
top10 <- read_csv("results/tables/thermo_nll_summary.csv", show_col_types=F) %>% head(10)
df_grid <- top10

calc_mets <- function(probs, y_true) {
    tryCatch({
        probs <- pmin(pmax(probs, 0.0001), 0.9999) # ensure valid probs
        df <- data.frame(truth=factor(y_true, levels=c("1", "0")), p1=probs)
        df$pred_class <- factor(ifelse(df$p1 > 0.5, "1", "0"), levels=c("1", "0"))
        pr <- pr_auc(df, truth, p1)$.estimate
        roc <- roc_auc(df, truth, p1)$.estimate
        m <- mcc(df, truth, pred_class)$.estimate
        brier <- brier_class(df, truth, p1)$.estimate
        c(pr, roc, m, brier)
    }, error=function(e) c(NA, NA, NA, NA))
}

run_base <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p) { v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res <- tryCatch(cma_es(rep(0,4), obj, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,4), value=NA))
        mets <- c(NA, NA, NA, NA)
        if(!is.na(res$value)) {
            e_bw <- ext_base(res$par, d$Boundary+1, d$Reward, d$RT)
            mets <- calc_mets(e_bw[,2], d$Boundary) # Index 2 is probability!
        }
        set.seed(s_idx)
        R_noise <- sample(c(0, 1), nrow(d), replace=TRUE)
        obj_rad <- function(p) { v <- get_nll_base(p, R_noise+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_rad <- tryCatch(cma_es(rep(0,4), obj_rad, control=list(maxit=30, sigma=0.5)), error=function(e) list(par=rep(0,4)))
        e_rad <- ext_base(res_rad$par, R_noise+1, d$Reward, d$RT)
        rad_cor <- abs(cor(e_rad[,2], R_noise))
        if(is.na(rad_cor)) rad_cor <- 0
        data.frame(SubjectID=s_idx, Base_NLL=res$value, Base_PR=mets[1], Base_ROC=mets[2], Base_MCC=mets[3], Base_Brier=mets[4], Base_Rad=rad_cor)
    }, error = function(e) data.frame(SubjectID=s_idx, Base_NLL=NA, Base_PR=NA, Base_ROC=NA, Base_MCC=NA, Base_Brier=NA, Base_Rad=NA))
}
cat("Evaluating Baseline...\n")
df_base <- bind_rows(mclapply(1:S, run_base, mc.cores=CORES))

run_cand <- function(s_idx, h) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p) { v <- get_nll_thermo_sudoku(p, h, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res <- tryCatch(cma_es(rep(0, 7), obj, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,7), value=NA))
        mets <- c(NA, NA, NA, NA)
        if(!is.na(res$value)) {
            ext <- ext_thermo_sudoku(res$par, h, d$Boundary+1, d$Reward, d$RT)
            mets <- calc_mets(ext[,2], d$Boundary)
        }
        set.seed(s_idx)
        R_noise <- sample(c(0, 1), nrow(d), replace=TRUE)
        obj_rad <- function(p) { v <- get_nll_thermo_sudoku(p, h, R_noise+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_rad <- tryCatch(cma_es(rep(0, 7), obj_rad, control=list(maxit=30, sigma=0.5)), error=function(e) list(par=rep(0,7)))
        ext_rad <- ext_thermo_sudoku(res_rad$par, h, R_noise+1, d$Reward, d$RT)
        rad_cor <- abs(cor(ext_rad[,2], R_noise))
        if(is.na(rad_cor)) rad_cor <- 0
        data.frame(SubjectID=s_idx, PR=mets[1], ROC=mets[2], MCC=mets[3], Brier=mets[4], Rad=rad_cor)
    }, error = function(e) data.frame(SubjectID=s_idx, PR=NA, ROC=NA, MCC=NA, Brier=NA, Rad=NA))
}

out_file <- "results/tables/thermo_final_stats.csv"
if(file.exists(out_file)) file.remove(out_file)
write_csv(data.frame(ModelID=character(), Mean_PR=numeric(), p_PR=numeric(), Mean_ROC=numeric(), p_ROC=numeric(), Mean_MCC=numeric(), p_MCC=numeric(), Mean_Brier=numeric(), p_Brier=numeric(), Mean_Rad=numeric(), p_Rad=numeric()), out_file)

cat("Evaluating Top 10 Candidates...\n")
for(i in 1:nrow(df_grid)) {
    mod_id <- df_grid$ModelID[i]
    h <- c(df_grid$struct_id[i], df_grid$kappa[i], df_grid$K_sa[i])
    df_cand <- bind_rows(mclapply(1:S, run_cand, h=h, mc.cores=CORES))
    df_merge <- inner_join(df_cand, df_base, by="SubjectID") %>% drop_na()
    
    if(nrow(df_merge) > 5) {
        safe_wt <- function(x, y) { if(sd(x)==0 && sd(y)==0) return(1.0); tryCatch(wilcox.test(x, y, paired=TRUE)$p.value, error=function(e) 1.0) }
        
        res_row <- data.frame(
            ModelID = mod_id,
            Mean_PR = mean(df_merge$PR), p_PR = safe_wt(df_merge$PR, df_merge$Base_PR),
            Mean_ROC = mean(df_merge$ROC), p_ROC = safe_wt(df_merge$ROC, df_merge$Base_ROC),
            Mean_MCC = mean(df_merge$MCC), p_MCC = safe_wt(df_merge$MCC, df_merge$Base_MCC),
            Mean_Brier = mean(df_merge$Brier), p_Brier = safe_wt(df_merge$Brier, df_merge$Base_Brier),
            Mean_Rad = mean(df_merge$Rad), p_Rad = safe_wt(df_merge$Rad, df_merge$Base_Rad)
        )
        write_csv(res_row, out_file, append=TRUE)
    }
}
cat("ALL METRICS EVALUATED.\n")
