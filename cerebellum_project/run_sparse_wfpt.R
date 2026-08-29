pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick, Metrics)
CORES <- parallel::detectCores()
Rcpp::sourceCpp("magi_sparse_wfpt.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=F)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0)) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

# Generate 50-model sparsity grid
# lambda_sparse (log-spaced from 1e-4 to 1e-1)
# K_sa (5 to 30)
l_grid <- 10^seq(-4, -1, length.out=10)
k_grid <- round(seq(5, 30, length.out=5))
grid <- expand.grid(lambda_sparse = l_grid, K_sa = k_grid)
grid$kappa <- 0.95
grid$ModelID <- sprintf("CC_SPARSE_%03d", 1:nrow(grid))

cat(sprintf("Total Sparse Models: %d\n", nrow(grid)))

calc_switch_mets <- function(probs, y_true) {
    tryCatch({
        probs <- pmin(pmax(probs, 0.0001), 0.9999)
        bound_prev <- c(NA, head(y_true, -1))
        switch_true <- abs(y_true - bound_prev)
        switch_prob <- abs(probs - bound_prev)
        
        df <- data.frame(truth=factor(switch_true[-1], levels=c("1", "0")), p1=switch_prob[-1])
        df$pred_class <- factor(ifelse(df$p1 > 0.5, "1", "0"), levels=c("1", "0"))
        
        pr <- pr_auc(df, truth, p1)$.estimate
        roc <- roc_auc(df, truth, p1)$.estimate
        m <- mcc(df, truth, pred_class)$.estimate
        c(pr, roc, m)
    }, error=function(e) c(NA,NA,NA))
}

cat("Phase 1: Evaluating Baseline WFPT (Switch Classification)...\n")
run_base <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p){v<-get_nll_base_w(p,d$Boundary+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res <- tryCatch(cma_es(rep(0,4),obj,control=list(maxit=50,sigma=0.5)),error=function(e)list(par=rep(0,4),value=NA))
        mets <- c(NA,NA,NA); rmse_v <- NA
        if(!is.na(res$value)){
            eb <- ext_base_w(res$par,d$Boundary+1,d$Reward,d$RT)
            mets <- calc_switch_mets(eb[,2],d$Boundary)
            rmse_v <- Metrics::rmse(d$RT, eb[,1])
        }
        set.seed(s_idx); R_noise <- sample(c(0,1),nrow(d),replace=TRUE)
        obj_r <- function(p){v<-get_nll_base_w(p,R_noise+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res_r <- tryCatch(cma_es(rep(0,4),obj_r,control=list(maxit=30,sigma=0.5)),error=function(e)list(par=rep(0,4)))
        er <- ext_base_w(res_r$par,R_noise+1,d$Reward,d$RT)
        rad <- abs(cor(er[,2],R_noise)); if(is.na(rad)) rad <- 0
        data.frame(SubjectID=s_idx,Base_NLL=res$value,Base_PR=mets[1],Base_ROC=mets[2],Base_MCC=mets[3],Base_Rad=rad,Base_RMSE=rmse_v)
    }, error=function(e) data.frame(SubjectID=s_idx,Base_NLL=NA,Base_PR=NA,Base_ROC=NA,Base_MCC=NA,Base_Rad=NA,Base_RMSE=NA))
}
df_base <- bind_rows(mclapply(1:S, run_base, mc.cores=CORES))

cat("Phase 2: Evaluating Sparse Variants...\n")
run_cand <- function(s_idx, h) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p){v<-get_nll_sparse(p,h,d$Boundary+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res <- tryCatch(cma_es(rep(0,7),obj,control=list(maxit=50,sigma=0.5)),error=function(e)list(par=rep(0,7),value=NA))
        mets <- c(NA,NA,NA); rmse_v <- NA
        if(!is.na(res$value)){
            ec <- ext_sparse(res$par,h,d$Boundary+1,d$Reward,d$RT)
            mets <- calc_switch_mets(ec[,2],d$Boundary)
            rmse_v <- Metrics::rmse(d$RT, ec[,1])
        }
        set.seed(s_idx); R_noise <- sample(c(0,1),nrow(d),replace=TRUE)
        obj_r <- function(p){v<-get_nll_sparse(p,h,R_noise+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res_r <- tryCatch(cma_es(rep(0,7),obj_r,control=list(maxit=30,sigma=0.5)),error=function(e)list(par=rep(0,7)))
        er <- ext_sparse(res_r$par,h,R_noise+1,d$Reward,d$RT)
        rad <- abs(cor(er[,2],R_noise)); if(is.na(rad)) rad <- 0
        data.frame(SubjectID=s_idx,NLL=res$value,PR=mets[1],ROC=mets[2],MCC=mets[3],Rad=rad,RMSE=rmse_v)
    }, error=function(e) data.frame(SubjectID=s_idx,NLL=NA,PR=NA,ROC=NA,MCC=NA,Rad=NA,RMSE=NA))
}

safe_wt <- function(x,y){if(length(x)<5||sd(x)==0&&sd(y)==0) return(1.0);tryCatch(wilcox.test(x,y,paired=TRUE)$p.value,error=function(e)1.0)}

out_file <- "results/tables/wfpt_sparse_terminal.csv"
if(file.exists(out_file)) file.remove(out_file)
header <- data.frame(ModelID=character(),Kappa=numeric(),Ksa=integer(),LambdaSparse=numeric(),
                     Mean_NLL=numeric(),p_NLL=numeric(),Mean_PR=numeric(),p_PR=numeric(),Mean_ROC=numeric(),p_ROC=numeric(),
                     Mean_MCC=numeric(),p_MCC=numeric(),Mean_RMSE=numeric(),p_RMSE=numeric(),
                     Mean_Rad=numeric(),p_Rad=numeric())
write_csv(header, out_file)

for(i in 1:nrow(grid)) {
    g <- grid[i,]
    h <- c(g$kappa, g$K_sa, g$lambda_sparse)
    cat(sprintf("[%d/%d] %s: K_sa=%d, LambdaSparse=%.2e\n", i, nrow(grid), g$ModelID, g$K_sa, g$lambda_sparse))
    
    df_cand <- bind_rows(mclapply(1:S, run_cand, h=h, mc.cores=CORES))
    df_m <- inner_join(df_cand, df_base, by="SubjectID") %>% drop_na()
    
    if(nrow(df_m)>5){
        row <- data.frame(
            ModelID=g$ModelID, Kappa=g$kappa, Ksa=g$K_sa, LambdaSparse=g$lambda_sparse,
            Mean_NLL=mean(df_m$NLL), p_NLL=safe_wt(df_m$NLL, df_m$Base_NLL),
            Mean_PR=mean(df_m$PR), p_PR=safe_wt(df_m$PR, df_m$Base_PR),
            Mean_ROC=mean(df_m$ROC), p_ROC=safe_wt(df_m$ROC, df_m$Base_ROC),
            Mean_MCC=mean(df_m$MCC), p_MCC=safe_wt(df_m$MCC, df_m$Base_MCC),
            Mean_RMSE=mean(df_m$RMSE), p_RMSE=safe_wt(df_m$RMSE, df_m$Base_RMSE),
            Mean_Rad=mean(df_m$Rad), p_Rad=safe_wt(df_m$Rad, df_m$Base_Rad)
        )
        write_csv(row, out_file, append=TRUE)
    }
}

cat("\n\n========== TERMINAL SUMMARY ==========\n")
df_final <- read_csv(out_file, show_col_types=F)
cat(sprintf("Models Evaluated: %d\n", nrow(df_final)))
cat(sprintf("Baseline Mean NLL: %.2f\n\n", mean(df_base$Base_NLL, na.rm=TRUE)))

# Find victors: ROC-AUC > Baseline (p < 0.05) AND Rademacher NOT worse (p_Rad > 0.05 or Mean_Rad <= Baseline_Rad)
# Wait, p_ROC < 0.05 and Mean_ROC > Baseline_ROC. Since we only have p-value, we check Mean_ROC and p_ROC.
baseline_mean_roc <- mean(df_base$Base_ROC, na.rm=TRUE)
victors <- df_final %>% filter(Mean_ROC > baseline_mean_roc, p_ROC < 0.05, p_Rad > 0.05) %>% arrange(desc(Mean_ROC))

if(nrow(victors) > 0) {
    cat("!!! VICTORIOUS ARCHITECTURES FOUND !!!\n")
    for(j in 1:nrow(victors)) {
        v <- victors[j,]
        cat(sprintf("  %s (L=%.2e, K=%d): ROC=%.3f (p=%.2e) | NLL=%.2f (p=%.2e) | PR(Switch)=%.3f | Rad=%.3f (p=%.2e)\n",
            v$ModelID, v$LambdaSparse, v$Ksa, v$Mean_ROC, v$p_ROC, v$Mean_NLL, v$p_NLL, v$Mean_PR, v$Mean_Rad, v$p_Rad))
    }
} else {
    cat("No models satisfied the Supremacy Threshold.\n")
}

cat("\nTOP 5 BY ROC-AUC (Regardless of Rademacher):\n")
best_roc <- df_final %>% arrange(desc(Mean_ROC)) %>% head(5)
for(j in 1:nrow(best_roc)) {
    b <- best_roc[j,]
    cat(sprintf("  %s (L=%.2e, K=%d): ROC=%.3f (p=%.2e) | NLL=%.2f (p=%.2e) | Rad=%.3f (p=%.2e)\n",
        b$ModelID, b$LambdaSparse, b$Ksa, b$Mean_ROC, b$p_ROC, b$Mean_NLL, b$p_NLL, b$Mean_Rad, b$p_Rad))
}
cat("======================================\n")
