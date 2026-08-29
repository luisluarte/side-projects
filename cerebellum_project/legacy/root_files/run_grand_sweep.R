pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick)

CORES <- parallel::detectCores()

Rcpp::sourceCpp("magi_grand_phylogeny.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    dplyr::mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% dplyr::filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    dplyr::mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

run_grand <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        y_true <- factor(d$Boundary, levels=c("1", "0"))
        
        calc_mets <- function(probs, rt_pred) {
            tryCatch({
                if(any(is.na(probs))) probs[is.na(probs)] <- 0.5
                df <- data.frame(truth = y_true, p1 = probs)
                df$pred_class <- factor(ifelse(df$p1 > 0.5, "1", "0"), levels=c("1", "0"))
                if(length(unique(d$Boundary)) < 2) return(c(PR_AUC=NA, ROC_AUC=NA, MCC=NA, Brier=NA, RT_RMSE=sqrt(mean((rt_pred - d$RT)^2))))
                pr <- pr_auc(df, truth, p1)$.estimate
                roc <- roc_auc(df, truth, p1)$.estimate
                m <- mcc(df, truth, pred_class)$.estimate
                brier <- brier_class(df, truth, p1)$.estimate
                rmse <- sqrt(mean((rt_pred - d$RT)^2))
                return(c(PR_AUC=pr, ROC_AUC=roc, MCC=m, Brier=brier, RT_RMSE=rmse))
            }, error = function(e) return(c(PR_AUC=NA, ROC_AUC=NA, MCC=NA, Brier=NA, RT_RMSE=NA)))
        }
        
        safe_cma <- function(fn, par_len) {
            res <- tryCatch(cma_es(rep(0, par_len), fn, control=list(maxit=100, sigma=0.5)), error = function(e) list(par=rep(0, par_len)))
            return(res$par)
        }
        
        # 1. Base DDM
        obj_ddm <- function(p) { v <- get_nll_ddm(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        p_ddm <- safe_cma(obj_ddm, 5)
        e_ddm <- ext_ddm(p_ddm, d$Boundary+1, d$Reward, d$RT); m_ddm <- calc_mets(e_ddm[,1], e_ddm[,0])
        
        # 2. Ctx DDM
        obj_ctx <- function(p) { v <- get_nll_ddm_ctx(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        p_ctx <- safe_cma(obj_ctx, 7)
        e_ctx <- ext_ddm_ctx(p_ctx, d$Boundary+1, d$Reward, d$RT); m_ctx <- calc_mets(e_ctx[,1], e_ctx[,0])
        
        # 3. Base Wald
        obj_bw <- function(p) { v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        p_bw <- safe_cma(obj_bw, 4)
        e_bw <- ext_base(p_bw, d$Boundary+1, d$Reward, d$RT); m_bw <- calc_mets(e_bw[,1], e_bw[,0])
        
        # 4. ECCM Reservoir
        obj_eccm <- function(p) { v <- get_nll_eccm(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        p_eccm <- safe_cma(obj_eccm, 6)
        e_eccm <- ext_eccm(p_eccm, d$Boundary+1, d$Reward, d$RT); m_eccm <- calc_mets(e_eccm[,1], e_eccm[,0])
        
        # 5. Q-Perturbed
        obj_qp <- function(p) { v <- get_nll_qperturbed(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        p_qp <- safe_cma(obj_qp, 10)
        e_qp <- ext_qperturbed(p_qp, d$Boundary+1, d$Reward, d$RT); m_qp <- calc_mets(e_qp[,1], e_qp[,0])
        
        # 6. Hybrid
        obj_hyb <- function(p) { v <- get_nll_hybrid(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        p_hyb <- safe_cma(obj_hyb, 12)
        e_hyb <- ext_hybrid(p_hyb, d$Boundary+1, d$Reward, d$RT); m_hyb <- calc_mets(e_hyb[,1], e_hyb[,0])
        
        return(data.frame(
            SubjectID = s_idx,
            Model = c("Baseline DDM", "Context DDM", "Baseline Wald", "ECCM Reservoir", "Q-Perturbed", "Terminal Hybrid"),
            PR_AUC = c(m_ddm[1], m_ctx[1], m_bw[1], m_eccm[1], m_qp[1], m_hyb[1]),
            ROC_AUC = c(m_ddm[2], m_ctx[2], m_bw[2], m_eccm[2], m_qp[2], m_hyb[2]),
            MCC = c(m_ddm[3], m_ctx[3], m_bw[3], m_eccm[3], m_qp[3], m_hyb[3]),
            Brier = c(m_ddm[4], m_ctx[4], m_bw[4], m_eccm[4], m_qp[4], m_hyb[4]),
            RT_RMSE = c(m_ddm[5], m_ctx[5], m_bw[5], m_eccm[5], m_qp[5], m_hyb[5])
        ))
    }, error = function(e) {
        return(data.frame(SubjectID=s_idx, Model=c("Baseline DDM", "Context DDM", "Baseline Wald", "ECCM Reservoir", "Q-Perturbed", "Terminal Hybrid"),
                          PR_AUC=NA, ROC_AUC=NA, MCC=NA, Brier=NA, RT_RMSE=NA))
    })
}

cat("Executing Grand Phylogeny NLL Sweep...\n")
res_list <- mclapply(1:S, run_grand, mc.cores = CORES)
df_mets <- bind_rows(res_list)

cat("Saving Metrics...\n")
write_csv(df_mets, "results/tables/grand_phylogeny_metrics.csv")

mets <- c("PR_AUC", "ROC_AUC", "MCC", "Brier", "RT_RMSE")
models <- c("Baseline DDM", "Context DDM", "ECCM Reservoir", "Q-Perturbed", "Terminal Hybrid")
base_model <- "Baseline Wald"

test_res <- list()
for(m in models) {
    for(met in mets) {
        df_base <- df_mets %>% filter(Model == base_model) %>% dplyr::select(SubjectID, BaseVal = !!sym(met))
        df_cand <- df_mets %>% filter(Model == m) %>% dplyr::select(SubjectID, CandVal = !!sym(met))
        df_paired <- inner_join(df_base, df_cand, by="SubjectID") %>% drop_na()
        
        if(nrow(df_paired) > 2) {
            val_base <- df_paired$BaseVal
            val_cand <- df_paired$CandVal
            
            wt <- wilcox.test(val_cand, val_base, paired = TRUE)
            mean_diff <- mean(val_cand - val_base)
            sd_diff <- sd(val_cand - val_base)
            cohens_d <- mean_diff / sd_diff
            mean_base <- mean(val_base)
            mean_cand <- mean(val_cand)
            
            test_res[[paste(m, met, sep="_")]] <- data.frame(
                Candidate = m, Metric = met,
                Mean_Baseline = mean_base, Mean_Candidate = mean_cand,
                Mean_Diff = mean_diff, P_Value = wt$p.value, Cohens_D = cohens_d
            )
        }
    }
}
df_stats <- bind_rows(test_res)

sink("results/tables/magi_grand_sweep_stats.txt")
cat("=== GRAND PHYLOGENY NLL SWEEP ===\n")
cat("Null Hypothesis (H0): Baseline Wald Architecture\n\n")
print(df_stats %>% dplyr::select(Candidate, Metric, Mean_Baseline, Mean_Candidate, Mean_Diff, P_Value, Cohens_D), row.names=FALSE)
sink()

cat("--- SWEEP COMPLETE ---\n")
