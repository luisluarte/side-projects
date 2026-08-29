library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 25
# 8 params: a, t_nd, beta_v, eta_LTP, eta_LTD, w_cb, lambda_shift, gamma_suppress
init_phi <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Inverse-Frequency Weighting...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 6: Unweighted Standard Fit (Cortical RPE architecture)
  chain_unweighted <- run_mcmc_subject(6, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_unweighted <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_unweighted[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 7: Weighted Fit (Cortical RPE architecture but objective function is inverse-freq weighted)
  chain_weighted <- run_mcmc_subject(7, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_weighted <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_weighted[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_unweighted <- numeric(nrow(p_data))
  pred_stay_weighted <- numeric(nrow(p_data))
  
  conf_unweighted_stay <- c()
  conf_unweighted_switch <- c()
  conf_weighted_stay <- c()
  conf_weighted_switch <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    true_ch <- p_data$Resp[i]
    prob_true_unweighted <- ifelse(true_ch == 1, metrics_unweighted$prob_ch1[i], 1 - metrics_unweighted$prob_ch1[i])
    prob_true_weighted <- ifelse(true_ch == 1, metrics_weighted$prob_ch1[i], 1 - metrics_weighted$prob_ch1[i])
    
    if (is_stay) {
      conf_unweighted_stay <- c(conf_unweighted_stay, prob_true_unweighted)
      conf_weighted_stay <- c(conf_weighted_stay, prob_true_weighted)
    } else {
      conf_unweighted_switch <- c(conf_unweighted_switch, prob_true_unweighted)
      conf_weighted_switch <- c(conf_weighted_switch, prob_true_weighted)
    }
    
    pred_ch_unweighted <- ifelse(metrics_unweighted$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_unweighted[i] <- ifelse(pred_ch_unweighted == p_data$Resp[i-1], 1, 0)
    
    pred_ch_weighted <- ifelse(metrics_weighted$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_weighted[i] <- ifelse(pred_ch_weighted == p_data$Resp[i-1], 1, 0)
  }
  
  TP_U <- sum(actual_stay[-1] == 1 & pred_stay_unweighted[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_U <- sum(actual_stay[-1] == 1 & pred_stay_unweighted[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_U <- sum(actual_stay[-1] == 0 & pred_stay_unweighted[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_U <- sum(actual_stay[-1] == 0 & pred_stay_unweighted[-1] == 0) / sum(actual_stay[-1] == 0)
  
  TP_W <- sum(actual_stay[-1] == 1 & pred_stay_weighted[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_W <- sum(actual_stay[-1] == 1 & pred_stay_weighted[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_W <- sum(actual_stay[-1] == 0 & pred_stay_weighted[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_W <- sum(actual_stay[-1] == 0 & pred_stay_weighted[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    TP_U = TP_U, FN_U = FN_U, FP_U = FP_U, TN_U = TN_U,
    TP_W = TP_W, FN_W = FN_W, FP_W = FP_W, TN_W = TN_W,
    conf_unweighted_stay = mean(conf_unweighted_stay, na.rm=TRUE),
    conf_unweighted_switch = mean(conf_unweighted_switch, na.rm=TRUE),
    conf_weighted_stay = mean(conf_weighted_stay, na.rm=TRUE),
    conf_weighted_switch = mean(conf_weighted_switch, na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Inverse-Frequency Weighted Fit: Results",
  "",
  "Comparing the Standard (Unweighted) MCMC fit against the Inverse-Frequency Weighted fit ($W_{switch} = N_{stay} / N_{switch}$). Both models use the Cortical RPE biological architecture.",
  "",
  "## 1. Weighted Fit Model",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_W), format_cell(results$FN_W)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_W), format_cell(results$TN_W)),
  "",
  "## 2. Standard Fit Model (Baseline)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_U), format_cell(results$FN_U)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_U), format_cell(results$TN_U)),
  "",
  "## 3. Confidence Change (Probability assigned to True Choice)",
  "| Behavior | Standard Fit | Weighted Fit |",
  "| :--- | :--- | :--- |",
  sprintf("| **Stay Trials** | %s | %s |", format_cell(results$conf_unweighted_stay), format_cell(results$conf_weighted_stay)),
  sprintf("| **Switch Trials** | %s | %s |", format_cell(results$conf_unweighted_switch), format_cell(results$conf_weighted_switch))
)

writeLines(report, "docs/Weighted_Fit_Results.md")
cat("\nReport successfully generated in docs/Weighted_Fit_Results.md\n")
