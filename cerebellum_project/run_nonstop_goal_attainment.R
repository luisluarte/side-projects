# ==============================================================================
# NON-STOP RECURSIVE ITERATION ENGINE FOR GOAL ATTAINMENT
# Target Criteria:
#   1. Continuous Reaction Time RMSE <= ~0.20 s
#   2. Switch PR-AUC >= ~0.70 (and Switch ROC-AUC >= 0.80)
#   3. Choice NLL <= 55.00
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(stats)
  library(PRROC)
  library(pROC)
  library(ggplot2)
})

cat("==============================================================================\n")
cat("STARTING NON-STOP ITERATION PIPELINE FOR STRICT GOAL ATTAINMENT\n")
cat("==============================================================================\n\n")

sourceCpp("ExactRModel.cpp")

dataset_path <- "../datasets/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

participants <- unique(dat_all[['participant_id']])
N_sub <- length(participants)

# Multi-parameter candidate sweeps
candidates <- list(
  list(name = "Targeted_Config_1", theta = c(0.910, 0.580, 0.35, 0.30, 0.80, 0.80, 0.45, 0.04, 0.03, 0.02)),
  list(name = "Targeted_Config_2", theta = c(0.920, 0.620, 0.45, 0.35, 1.20, 1.20, 0.50, 0.05, 0.04, 0.03)),
  list(name = "Targeted_Config_3", theta = c(0.930, 0.650, 0.55, 0.38, 1.60, 1.60, 0.55, 0.06, 0.05, 0.04)),
  list(name = "Targeted_Config_4", theta = c(0.940, 0.700, 0.65, 0.40, 2.00, 2.00, 0.60, 0.06, 0.05, 0.05)),
  list(name = "Targeted_Config_5", theta = c(0.950, 0.750, 0.80, 0.42, 2.50, 2.50, 0.65, 0.07, 0.06, 0.05))
)

eval_candidate <- function(th) {
  all_lbls <- numeric(0); all_prbs <- numeric(0)
  all_rt_e <- numeric(0); all_rt_p <- numeric(0)
  all_val  <- numeric(0); all_unc  <- numeric(0)
  sub_nll  <- numeric(N_sub)
  
  for (s in 1:N_sub) {
    sub_df <- dat_all[dat_all[['participant_id']] == participants[s], ]
    res <- run_exact_r_simulation_cpp(
      as.numeric(sub_df[['Resp']]),
      as.numeric(sub_df[['F']]),
      as.numeric(sub_df[['Bd1']]),
      as.numeric(sub_df[['Bd2']]),
      as.numeric(sub_df[['RT']]),
      th
    )
    sub_nll[s] <- res$Choice_NLL
    all_lbls <- c(all_lbls, as.numeric(res$Switch_Labels))
    all_prbs <- c(all_prbs, as.numeric(res$Switch_Probs))
    all_rt_e <- c(all_rt_e, as.numeric(res$RT_Emp))
    all_rt_p <- c(all_rt_p, as.numeric(res$RT_Preds))
    all_val  <- c(all_val,  as.numeric(res$Value_Traj))
    all_unc  <- c(all_unc,  as.numeric(res$Uncertainty_Traj))
  }
  
  clean_sw <- !is.na(all_lbls) & !is.na(all_prbs)
  pr_res <- pr.curve(scores.class0 = all_prbs[clean_sw & all_lbls == 1],
                     scores.class1 = all_prbs[clean_sw & all_lbls == 0], curve = FALSE)
  roc_res <- as.numeric(pROC::auc(all_lbls[clean_sw], all_prbs[clean_sw], quiet = TRUE))
  
  clean_rt <- !is.na(all_rt_e) & !is.na(all_rt_p)
  rt_rmse <- sqrt(mean((all_rt_e[clean_rt] - all_rt_p[clean_rt])^2))
  rt_r2 <- 1.0 - sum((all_rt_e[clean_rt] - all_rt_p[clean_rt])^2) / sum((all_rt_e[clean_rt] - mean(all_rt_e[clean_rt]))^2)
  
  return(list(
    Choice_NLL = mean(sub_nll),
    PR_AUC = pr_res[['auc.integral']],
    ROC_AUC = roc_res,
    RT_RMSE = rt_rmse,
    RT_R2 = rt_r2,
    all_val = all_val,
    all_unc = all_unc,
    all_rt_e = all_rt_e,
    all_rt_p = all_rt_p
  ))
}

for (i in 1:length(candidates)) {
  c_res <- eval_candidate(candidates[[i]]$theta)
  cat(sprintf("Iterative Evaluation %d [%s]:\n", i, candidates[[i]]$name))
  cat(sprintf("  - Choice NLL:    %.4f\n", c_res$Choice_NLL))
  cat(sprintf("  - Switch ROC-AUC: %.4f\n", c_res$ROC_AUC))
  cat(sprintf("  - Switch PR-AUC:  %.4f\n", c_res$PR_AUC))
  cat(sprintf("  - RT RMSE:        %.4f s\n", c_res$RT_RMSE))
  cat(sprintf("  - RT R^2:         %.4f\n\n", c_res$RT_R2))
}
