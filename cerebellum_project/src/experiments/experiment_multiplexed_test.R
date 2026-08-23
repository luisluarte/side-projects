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

cat("Running parallel MCMC fits to test Multiplexed Meta-Learning...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  sourceCpp("src/models/evaluate_metrics_multiplexed.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 6: Standard Cortical RPE Feedback
  chain_baseline <- run_mcmc_subject(6, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_baseline <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_baseline[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_baseline <- eval_eccm_cortical_rpe(as.numeric(chain_baseline[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 8: Multiplexed Meta-Learning
  chain_multi <- run_mcmc_subject(8, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_multi <- eval_metrics_eccm_multiplexed(as.numeric(chain_multi[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_multi <- eval_eccm_multiplexed(as.numeric(chain_multi[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_baseline <- numeric(nrow(p_data))
  pred_stay_multi <- numeric(nrow(p_data))
  
  conf_base_stay <- c()
  conf_base_switch <- c()
  conf_multi_stay <- c()
  conf_multi_switch <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    true_ch <- p_data$Resp[i]
    prob_true_base <- ifelse(true_ch == 1, metrics_baseline$prob_ch1[i], 1 - metrics_baseline$prob_ch1[i])
    prob_true_multi <- ifelse(true_ch == 1, metrics_multi$prob_ch1[i], 1 - metrics_multi$prob_ch1[i])
    
    if (is_stay) {
      conf_base_stay <- c(conf_base_stay, prob_true_base)
      conf_multi_stay <- c(conf_multi_stay, prob_true_multi)
    } else {
      conf_base_switch <- c(conf_base_switch, prob_true_base)
      conf_multi_switch <- c(conf_multi_switch, prob_true_multi)
    }
    
    pred_ch_base <- ifelse(metrics_baseline$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_baseline[i] <- ifelse(pred_ch_base == p_data$Resp[i-1], 1, 0)
    
    pred_ch_multi <- ifelse(metrics_multi$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_multi[i] <- ifelse(pred_ch_multi == p_data$Resp[i-1], 1, 0)
  }
  
  TP_B <- sum(actual_stay[-1] == 1 & pred_stay_baseline[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_B <- sum(actual_stay[-1] == 1 & pred_stay_baseline[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_B <- sum(actual_stay[-1] == 0 & pred_stay_baseline[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_B <- sum(actual_stay[-1] == 0 & pred_stay_baseline[-1] == 0) / sum(actual_stay[-1] == 0)
  
  TP_M <- sum(actual_stay[-1] == 1 & pred_stay_multi[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_M <- sum(actual_stay[-1] == 1 & pred_stay_multi[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_M <- sum(actual_stay[-1] == 0 & pred_stay_multi[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_M <- sum(actual_stay[-1] == 0 & pred_stay_multi[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    dev_baseline = dev_baseline, dev_multi = dev_multi,
    TP_B = TP_B, FN_B = FN_B, FP_B = FP_B, TN_B = TN_B,
    TP_M = TP_M, FN_M = FN_M, FP_M = FP_M, TN_M = TN_M,
    conf_base_stay = mean(conf_base_stay, na.rm=TRUE),
    conf_base_switch = mean(conf_base_switch, na.rm=TRUE),
    conf_multi_stay = mean(conf_multi_stay, na.rm=TRUE),
    conf_multi_switch = mean(conf_multi_switch, na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Multiplexed Meta-Learning Architecture: Results",
  "",
  "Comparing the **Cortical RPE** baseline against the new **Multiplexed** model where the Cerebellum explicitly outputs Stay and Switch meta-values rather than Target-specific Q-values.",
  "",
  "## 1. Deviance Check",
  sprintf("Mean Deviance Baseline (Cortical RPE): %.2f", mean(results$dev_baseline, na.rm=TRUE)),
  sprintf("Mean Deviance Multiplexed: %.2f", mean(results$dev_multi, na.rm=TRUE)),
  sprintf("Paired t-test p-value: %.4e", t.test(results$dev_baseline, results$dev_multi, paired=TRUE, alternative="greater")$p.value),
  "",
  "## 2. Multiplexed Architecture (New Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_M), format_cell(results$FN_M)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_M), format_cell(results$TN_M)),
  "",
  "## 3. Standard Target Architecture (Baseline)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_B), format_cell(results$FN_B)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_B), format_cell(results$TN_B)),
  "",
  "## 4. Confidence Change (Probability assigned to True Choice)",
  "| Behavior | Standard Target Output | Multiplexed Stay/Switch Output |",
  "| :--- | :--- | :--- |",
  sprintf("| **Stay Trials** | %s | %s |", format_cell(results$conf_base_stay), format_cell(results$conf_multi_stay)),
  sprintf("| **Switch Trials** | %s | %s |", format_cell(results$conf_base_switch), format_cell(results$conf_multi_switch))
)

writeLines(report, "docs/Multiplexed_Results.md")
cat("\nReport successfully generated in docs/Multiplexed_Results.md\n")
