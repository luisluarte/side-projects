pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick, Metrics)
CORES <- parallel::detectCores()
Rcpp::sourceCpp("magi_100_wfpt.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=F)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0)) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

# Generate 100-model grid
# Axis1: expansion (1=delay,2=reservoir,3=fractional) x
# Axis2: trace (1=mono,2=dual) x
# Axis3: mask (1=indep,2=modular) x
# Axis4: integration (1=drift,2=boundary)
# = 3x2x2x2 = 24 structural configs
# Parametric: kappa in {0.3, 0.5, 0.8, 0.95}, K_sa in {5, 10, 20}
# 24 x 4 kappas = 96 (pick K_sa=10 default)
# + 4 extra with best struct at K_sa=5,20,30,50

grid_base <- expand.grid(
    exp_type = 1:3,
    trace_type = 1:2,
    mask_type = 1:2,
    integ_type = 1:2,
    kappa = c(0.3, 0.5, 0.8, 0.95),
    K_sa = 10
)
# Add 4 extras with varied K_sa
grid_extra <- data.frame(
    exp_type = c(1,2,3,1),
    trace_type = c(1,1,2,2),
    mask_type = c(1,2,1,2),
    integ_type = c(1,1,2,2),
    kappa = c(0.8, 0.8, 0.8, 0.8),
    K_sa = c(5, 20, 30, 50)
)
grid <- bind_rows(grid_base, grid_extra)
grid$ModelID <- sprintf("CC_WFPT_%03d", 1:nrow(grid))
cat(sprintf("Total models: %d\n", nrow(grid)))

calc_mets <- function(probs, y_true) {
    tryCatch({
        probs <- pmin(pmax(probs, 0.0001), 0.9999)
        df <- data.frame(truth=factor(y_true, levels=c("1","0")), p1=probs)
        df$pred_class <- factor(ifelse(df$p1>0.5,"1","0"), levels=c("1","0"))
        c(pr_auc(df,truth,p1)$.estimate, roc_auc(df,truth,p1)$.estimate,
          mcc(df,truth,pred_class)$.estimate, brier_class(df,truth,p1)$.estimate)
    }, error=function(e) c(NA,NA,NA,NA))
}

# Phase 1: Baseline (once)
cat("Phase 1: Evaluating Baseline WFPT...\n")
run_base <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p){v<-get_nll_base_w(p,d$Boundary+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res <- tryCatch(cma_es(rep(0,4),obj,control=list(maxit=50,sigma=0.5)),error=function(e)list(par=rep(0,4),value=NA))
        mets <- c(NA,NA,NA,NA); rmse_v <- NA
        if(!is.na(res$value)){
            eb <- ext_base_w(res$par,d$Boundary+1,d$Reward,d$RT)
            mets <- calc_mets(eb[,2],d$Boundary)
            rmse_v <- Metrics::rmse(d$RT, eb[,1])
        }
        set.seed(s_idx); R_noise <- sample(c(0,1),nrow(d),replace=TRUE)
        obj_r <- function(p){v<-get_nll_base_w(p,R_noise+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res_r <- tryCatch(cma_es(rep(0,4),obj_r,control=list(maxit=30,sigma=0.5)),error=function(e)list(par=rep(0,4)))
        er <- ext_base_w(res_r$par,R_noise+1,d$Reward,d$RT)
        rad <- abs(cor(er[,2],R_noise)); if(is.na(rad)) rad <- 0
        data.frame(SubjectID=s_idx,Base_NLL=res$value,Base_PR=mets[1],Base_ROC=mets[2],Base_MCC=mets[3],Base_Brier=mets[4],Base_Rad=rad,Base_RMSE=rmse_v)
    }, error=function(e) data.frame(SubjectID=s_idx,Base_NLL=NA,Base_PR=NA,Base_ROC=NA,Base_MCC=NA,Base_Brier=NA,Base_Rad=NA,Base_RMSE=NA))
}
df_base <- bind_rows(mclapply(1:S, run_base, mc.cores=CORES))

# Phase 2: All 100 models
cat("Phase 2: Evaluating 100 Structural Candidates...\n")
run_cand <- function(s_idx, h) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p){v<-get_nll_100(p,h,d$Boundary+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res <- tryCatch(cma_es(rep(0,7),obj,control=list(maxit=50,sigma=0.5)),error=function(e)list(par=rep(0,7),value=NA))
        mets <- c(NA,NA,NA,NA); rmse_v <- NA
        if(!is.na(res$value)){
            ec <- ext_100(res$par,h,d$Boundary+1,d$Reward,d$RT)
            mets <- calc_mets(ec[,2],d$Boundary)
            rmse_v <- Metrics::rmse(d$RT, ec[,1])
        }
        set.seed(s_idx); R_noise <- sample(c(0,1),nrow(d),replace=TRUE)
        obj_r <- function(p){v<-get_nll_100(p,h,R_noise+1,d$Reward,d$RT);if(is.nan(v))1e6 else v}
        res_r <- tryCatch(cma_es(rep(0,7),obj_r,control=list(maxit=30,sigma=0.5)),error=function(e)list(par=rep(0,7)))
        er <- ext_100(res_r$par,h,R_noise+1,d$Reward,d$RT)
        rad <- abs(cor(er[,2],R_noise)); if(is.na(rad)) rad <- 0
        data.frame(SubjectID=s_idx,NLL=res$value,PR=mets[1],ROC=mets[2],MCC=mets[3],Brier=mets[4],Rad=rad,RMSE=rmse_v)
    }, error=function(e) data.frame(SubjectID=s_idx,NLL=NA,PR=NA,ROC=NA,MCC=NA,Brier=NA,Rad=NA,RMSE=NA))
}

safe_wt <- function(x,y){if(length(x)<5||sd(x)==0&&sd(y)==0) return(1.0);tryCatch(wilcox.test(x,y,paired=TRUE)$p.value,error=function(e)1.0)}

out_file <- "results/tables/wfpt_100_terminal.csv"
if(file.exists(out_file)) file.remove(out_file)
header <- data.frame(ModelID=character(),Exp=integer(),Trace=integer(),Mask=integer(),Integ=integer(),Kappa=numeric(),Ksa=integer(),
                     Mean_NLL=numeric(),p_NLL=numeric(),Mean_PR=numeric(),p_PR=numeric(),Mean_ROC=numeric(),p_ROC=numeric(),
                     Mean_MCC=numeric(),p_MCC=numeric(),Mean_Brier=numeric(),p_Brier=numeric(),Mean_RMSE=numeric(),p_RMSE=numeric(),
                     Mean_Rad=numeric(),p_Rad=numeric())
write_csv(header, out_file)

exp_names <- c("DelayLine","Reservoir","Fractional")
trace_names <- c("MonoExp","DualCascade")
mask_names <- c("IndepNode","ModularMZ")
integ_names <- c("DriftGate","BoundaryExp")

for(i in 1:nrow(grid)) {
    g <- grid[i,]
    h <- c(g$exp_type, g$trace_type, g$mask_type, g$integ_type, g$kappa, g$K_sa)
    cat(sprintf("[%d/%d] %s: %s/%s/%s/%s kappa=%.2f K_sa=%d\n", i, nrow(grid), g$ModelID,
        exp_names[g$exp_type], trace_names[g$trace_type], mask_names[g$mask_type], integ_names[g$integ_type], g$kappa, g$K_sa))
    
    df_cand <- bind_rows(mclapply(1:S, run_cand, h=h, mc.cores=CORES))
    df_m <- inner_join(df_cand, df_base, by="SubjectID") %>% drop_na()
    
    if(nrow(df_m)>5){
        row <- data.frame(
            ModelID=g$ModelID, Exp=g$exp_type, Trace=g$trace_type, Mask=g$mask_type, Integ=g$integ_type, Kappa=g$kappa, Ksa=g$K_sa,
            Mean_NLL=mean(df_m$NLL), p_NLL=safe_wt(df_m$NLL, df_m$Base_NLL),
            Mean_PR=mean(df_m$PR), p_PR=safe_wt(df_m$PR, df_m$Base_PR),
            Mean_ROC=mean(df_m$ROC), p_ROC=safe_wt(df_m$ROC, df_m$Base_ROC),
            Mean_MCC=mean(df_m$MCC), p_MCC=safe_wt(df_m$MCC, df_m$Base_MCC),
            Mean_Brier=mean(df_m$Brier), p_Brier=safe_wt(df_m$Brier, df_m$Base_Brier),
            Mean_RMSE=mean(df_m$RMSE), p_RMSE=safe_wt(df_m$RMSE, df_m$Base_RMSE),
            Mean_Rad=mean(df_m$Rad), p_Rad=safe_wt(df_m$Rad, df_m$Base_Rad)
        )
        write_csv(row, out_file, append=TRUE)
    }
}

# Print summary
cat("\n\n========== TERMINAL SUMMARY ==========\n")
df_final <- read_csv(out_file, show_col_types=F)
cat(sprintf("Models Evaluated: %d\n", nrow(df_final)))
cat(sprintf("Baseline Mean NLL: %.2f\n\n", mean(df_base$Base_NLL, na.rm=TRUE)))

best_nll <- df_final %>% arrange(Mean_NLL) %>% head(5)
cat("TOP 5 BY NLL:\n")
for(j in 1:nrow(best_nll)) {
    b <- best_nll[j,]
    cat(sprintf("  %s: NLL=%.2f (p=%.2e) | PR=%.3f (p=%.2e) | ROC=%.3f (p=%.2e) | MCC=%.3f (p=%.2e) | Brier=%.3f (p=%.2e) | RMSE=%.3f (p=%.2e) | Rad=%.3f (p=%.2e)\n",
        b$ModelID, b$Mean_NLL, b$p_NLL, b$Mean_PR, b$p_PR, b$Mean_ROC, b$p_ROC, b$Mean_MCC, b$p_MCC, b$Mean_Brier, b$p_Brier, b$Mean_RMSE, b$p_RMSE, b$Mean_Rad, b$p_Rad))
}

best_roc <- df_final %>% arrange(desc(Mean_ROC)) %>% head(5)
cat("\nTOP 5 BY ROC-AUC:\n")
for(j in 1:nrow(best_roc)) {
    b <- best_roc[j,]
    cat(sprintf("  %s: ROC=%.3f (p=%.2e) | NLL=%.2f (p=%.2e) | MCC=%.3f | RMSE=%.3f | Rad=%.3f\n",
        b$ModelID, b$Mean_ROC, b$p_ROC, b$Mean_NLL, b$p_NLL, b$Mean_MCC, b$Mean_RMSE, b$Mean_Rad))
}
cat("======================================\n")
