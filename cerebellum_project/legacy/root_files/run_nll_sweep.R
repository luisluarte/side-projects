pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick)

CORES <- parallel::detectCores()
hyper <- c(0.01, 1.00, 2.0)

Rcpp::sourceCpp("magi_nll_sweep.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    dplyr::mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% dplyr::filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    dplyr::mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

run_nll <- function(s_idx) {
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
        
        obj_b <- function(p) { v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        par_b <- safe_cma(obj_b, 4)
        ext_b <- ext_base(par_b, d$Boundary+1, d$Reward, d$RT)
        m_b <- calc_mets(ext_b[,2], ext_b[,1])
        
        obj_q <- function(p) { v <- get_nll_qperturbed(p, hyper, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        par_q <- safe_cma(obj_q, 10)
        ext_q <- ext_qperturbed(par_q, hyper, d$Boundary+1, d$Reward, d$RT)
        m_q <- calc_mets(ext_q[,2], ext_q[,1])
        
        obj_h <- function(p) { v <- get_nll_hybrid(p, hyper, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) return(1e6) else return(v) }
        par_h <- safe_cma(obj_h, 12)
        ext_h <- ext_hybrid(par_h, hyper, d$Boundary+1, d$Reward, d$RT)
        m_h <- calc_mets(ext_h[,2], ext_h[,1])
        
        return(data.frame(
            SubjectID = s_idx,
            Model = c("Baseline Wald", "Q-Perturbed (LTI)", "Terminal Hybrid"),
            PR_AUC = c(m_b[1], m_q[1], m_h[1]),
            ROC_AUC = c(m_b[2], m_q[2], m_h[2]),
            MCC = c(m_b[3], m_q[3], m_h[3]),
            Brier = c(m_b[4], m_q[4], m_h[4]),
            RT_RMSE = c(m_b[5], m_q[5], m_h[5])
        ))
    }, error = function(e) {
        return(data.frame(SubjectID=s_idx, Model=c("Baseline Wald", "Q-Perturbed (LTI)", "Terminal Hybrid"),
                          PR_AUC=NA, ROC_AUC=NA, MCC=NA, Brier=NA, RT_RMSE=NA))
    })
}

cat("Executing Maximum Likelihood NLL Parameter Sweep...\n")
res_list <- mclapply(1:S, run_nll, mc.cores = CORES)
df_mets <- bind_rows(res_list)

cat("Performing Statistical Supremacy Testing...\n")

mets <- c("PR_AUC", "ROC_AUC", "MCC", "Brier", "RT_RMSE")
models <- c("Q-Perturbed (LTI)", "Terminal Hybrid")
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

sink("results/tables/magi_nll_sweep_stats.txt")
cat("=== MAXIMUM LIKELIHOOD PHYLOGENY SWEEP (NLL) ===\n")
cat("Null Hypothesis (H0): Baseline Wald Architecture\n\n")

print(df_stats %>% dplyr::select(Candidate, Metric, Mean_Baseline, Mean_Candidate, Mean_Diff, P_Value, Cohens_D), row.names=FALSE)
sink()

cat("--- SWEEP COMPLETE ---\n")
