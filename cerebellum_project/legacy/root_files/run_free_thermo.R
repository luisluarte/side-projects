pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick)
CORES <- parallel::detectCores()

Rcpp::sourceCpp("magi_thermo_sudoku_free.cpp")
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

calc_mets <- function(probs, y_true) {
    tryCatch({
        probs <- pmin(pmax(probs, 0.0001), 0.9999)
        df <- data.frame(truth=factor(y_true, levels=c("1", "0")), p1=probs)
        df$pred_class <- factor(ifelse(df$p1 > 0.5, "1", "0"), levels=c("1", "0"))
        pr <- pr_auc(df, truth, p1)$.estimate
        roc <- roc_auc(df, truth, p1)$.estimate
        m <- mcc(df, truth, pred_class)$.estimate
        brier <- brier_class(df, truth, p1)$.estimate
        c(pr, roc, m, brier)
    }, error=function(e) c(NA, NA, NA, NA))
}

run_free <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        
        # Base Wald
        obj_base <- function(p) { v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_base <- tryCatch(cma_es(rep(0,4), obj_base, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,4), value=NA))
        mets_base <- c(NA, NA, NA, NA)
        if(!is.na(res_base$value)) {
            e_bw <- ext_base(res_base$par, d$Boundary+1, d$Reward, d$RT)
            mets_base <- calc_mets(e_bw[,2], d$Boundary)
        }
        set.seed(s_idx)
        R_noise <- sample(c(0, 1), nrow(d), replace=TRUE)
        obj_rad_base <- function(p) { v <- get_nll_base(p, R_noise+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_rad_base <- tryCatch(cma_es(rep(0,4), obj_rad_base, control=list(maxit=30, sigma=0.5)), error=function(e) list(par=rep(0,4)))
        e_rad_base <- ext_base(res_rad_base$par, R_noise+1, d$Reward, d$RT)
        rad_base <- abs(cor(e_rad_base[,2], R_noise))
        if(is.na(rad_base)) rad_base <- 0
        
        # Free CC_Model_069
        obj_cand <- function(p) { v <- get_nll_thermo_sudoku_free(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_cand <- tryCatch(cma_es(rep(0, 9), obj_cand, control=list(maxit=50, sigma=0.5)), error=function(e) list(par=rep(0,9), value=NA))
        mets_cand <- c(NA, NA, NA, NA)
        if(!is.na(res_cand$value)) {
            e_cand <- ext_thermo_sudoku_free(res_cand$par, d$Boundary+1, d$Reward, d$RT)
            mets_cand <- calc_mets(e_cand[,2], d$Boundary)
        }
        obj_rad_cand <- function(p) { v <- get_nll_thermo_sudoku_free(p, R_noise+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res_rad_cand <- tryCatch(cma_es(rep(0, 9), obj_rad_cand, control=list(maxit=30, sigma=0.5)), error=function(e) list(par=rep(0,9)))
        e_rad_cand <- ext_thermo_sudoku_free(res_rad_cand$par, R_noise+1, d$Reward, d$RT)
        rad_cand <- abs(cor(e_rad_cand[,2], R_noise))
        if(is.na(rad_cand)) rad_cand <- 0
        
        # Kappa and K_sa decoding
        phi_kappa <- res_cand$par[8]
        phi_ksa <- res_cand$par[9]
        k_val <- 1.0 / (1.0 + exp(-phi_kappa))
        ksa_val <- round(50.0 / (1.0 + exp(-phi_ksa))) + 1
        
        data.frame(
            SubjectID=s_idx, 
            Base_NLL=res_base$value, Base_PR=mets_base[1], Base_ROC=mets_base[2], Base_MCC=mets_base[3], Base_Brier=mets_base[4], Base_Rad=rad_base,
            Cand_NLL=res_cand$value, Cand_PR=mets_cand[1], Cand_ROC=mets_cand[2], Cand_MCC=mets_cand[3], Cand_Brier=mets_cand[4], Cand_Rad=rad_cand,
            Kappa_Fit=k_val, Ksa_Fit=ksa_val
        )
    }, error = function(e) data.frame(SubjectID=s_idx, Base_NLL=NA, Base_PR=NA, Base_ROC=NA, Base_MCC=NA, Base_Brier=NA, Base_Rad=NA,
                                      Cand_NLL=NA, Cand_PR=NA, Cand_ROC=NA, Cand_MCC=NA, Cand_Brier=NA, Cand_Rad=NA, Kappa_Fit=NA, Ksa_Fit=NA))
}

cat("Evaluating Free-Parameter Thermo Model...\n")
df_res <- bind_rows(mclapply(1:S, run_free, mc.cores=CORES)) %>% drop_na()

safe_wt <- function(x, y) { if(sd(x)==0 && sd(y)==0) return(1.0); tryCatch(wilcox.test(x, y, paired=TRUE)$p.value, error=function(e) 1.0) }

cat("\n======================================\n")
cat("CC_Model_069 (Free kappa/K_sa) vs Baseline\n")
cat("======================================\n")
cat(sprintf("NLL        : %.2f vs %.2f (p = %.2e)\n", mean(df_res$Cand_NLL), mean(df_res$Base_NLL), safe_wt(df_res$Cand_NLL, df_res$Base_NLL)))
cat(sprintf("PR-AUC     : %.3f vs %.3f (p = %.2e)\n", mean(df_res$Cand_PR), mean(df_res$Base_PR), safe_wt(df_res$Cand_PR, df_res$Base_PR)))
cat(sprintf("ROC-AUC    : %.3f vs %.3f (p = %.2e)\n", mean(df_res$Cand_ROC), mean(df_res$Base_ROC), safe_wt(df_res$Cand_ROC, df_res$Base_ROC)))
cat(sprintf("MCC        : %.3f vs %.3f (p = %.2e)\n", mean(df_res$Cand_MCC), mean(df_res$Base_MCC), safe_wt(df_res$Cand_MCC, df_res$Base_MCC)))
cat(sprintf("Brier      : %.3f vs %.3f (p = %.2e)\n", mean(df_res$Cand_Brier), mean(df_res$Base_Brier), safe_wt(df_res$Cand_Brier, df_res$Base_Brier)))
cat(sprintf("Rademacher : %.3f vs %.3f (p = %.2e)\n", mean(df_res$Cand_Rad), mean(df_res$Base_Rad), safe_wt(df_res$Cand_Rad, df_res$Base_Rad)))
cat("--------------------------------------\n")
cat(sprintf("Fitted Kappa (mean) : %.3f (sd = %.3f)\n", mean(df_res$Kappa_Fit), sd(df_res$Kappa_Fit)))
cat(sprintf("Fitted K_sa  (mean) : %.1f (sd = %.1f)\n", mean(df_res$Ksa_Fit), sd(df_res$Ksa_Fit)))
cat("======================================\n")
