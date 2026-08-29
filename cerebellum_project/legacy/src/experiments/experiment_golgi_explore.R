library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 75
# Model 12: 9 params
init_phi_12 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1))
# Model 17: 11 params (adds mf_threshold, explore_gain)
init_phi_17 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), log(1.0), log(0.5))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits for Model 17: MF-Gated Exploration...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi_explore.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 12 (Baseline Golgi Tanh Divisive)
  chain_12 <- run_mcmc_subject(12, iters, init_phi_12, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_12 <- eval_metrics_eccm_golgi(as.numeric(chain_12[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_12 <- eval_eccm_golgi(as.numeric(chain_12[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 17 (MF-Gated Exploration)
  chain_17 <- run_mcmc_subject(17, iters, init_phi_17, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_17 <- eval_metrics_eccm_golgi_explore(as.numeric(chain_17[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_17 <- eval_eccm_golgi_explore(as.numeric(chain_17[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  pred_switch_12 <- 0
  pred_switch_17 <- 0
  actual_switches <- 0
  conf_switch_12 <- c()
  conf_switch_17 <- c()
  
  # Full confusion matrix for Model 17
  actual_stay <- numeric(nrow(p_data))
  pred_stay_17 <- numeric(nrow(p_data))
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    pred_ch_17 <- ifelse(metrics_17$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_17[i] <- ifelse(pred_ch_17 == p_data$Resp[i-1], 1, 0)
    
    if (!is_stay) {
      actual_switches <- actual_switches + 1
      if (ifelse(metrics_12$prob_ch1[i] > 0.5, 1, 2) != p_data$Resp[i-1]) pred_switch_12 <- pred_switch_12 + 1
      if (ifelse(metrics_17$prob_ch1[i] > 0.5, 1, 2) != p_data$Resp[i-1]) pred_switch_17 <- pred_switch_17 + 1
      
      conf_switch_12 <- c(conf_switch_12, ifelse(p_data$Resp[i]==1, metrics_12$prob_ch1[i], 1.0 - metrics_12$prob_ch1[i]))
      conf_switch_17 <- c(conf_switch_17, ifelse(p_data$Resp[i]==1, metrics_17$prob_ch1[i], 1.0 - metrics_17$prob_ch1[i]))
    }
  }
  
  TP <- sum(actual_stay[-1] == 1 & pred_stay_17[-1] == 1) / max(1, sum(actual_stay[-1] == 1))
  FN <- sum(actual_stay[-1] == 1 & pred_stay_17[-1] == 0) / max(1, sum(actual_stay[-1] == 1))
  FP <- sum(actual_stay[-1] == 0 & pred_stay_17[-1] == 1) / max(1, sum(actual_stay[-1] == 0))
  TN <- sum(actual_stay[-1] == 0 & pred_stay_17[-1] == 0) / max(1, sum(actual_stay[-1] == 0))
  
  data.frame(
    subject = p,
    switches = actual_switches,
    pred_switch_12 = pred_switch_12,
    pred_switch_17 = pred_switch_17,
    dev_12 = dev_12, dev_17 = dev_17,
    TP = TP, FN = FN, FP = FP, TN = TN,
    conf_12 = mean(conf_switch_12, na.rm=TRUE),
    conf_17 = mean(conf_switch_17, na.rm=TRUE),
    theta_mf = exp(chain_17[iters, 10]),
    theta_explore = exp(chain_17[iters, 11])
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

sum_switches <- sum(results$switches)
recall_12 <- sum(results$pred_switch_12) / sum_switches * 100
recall_17 <- sum(results$pred_switch_17) / sum_switches * 100

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Model 17: MF-Gated Exploration (Golgi + Counter-Drift)",
  "",
  "Uses ΔMF Energy (the proven strongest switch predictor from brute-force analysis)",
  "to inject a counter-drift that pushes AGAINST the current choice when MF energy",
  "spikes above a learned threshold.",
  "",
  "## Switch Recall (Target: >55%)",
  sprintf("*   **Model 12 (Baseline Golgi Tanh):** %.2f%%", recall_12),
  sprintf("*   **Model 17 (MF-Gated Exploration):** %.2f%%", recall_17),
  sprintf("*   **Delta:** %+.2f%%", recall_17 - recall_12),
  "",
  "## Deviance",
  sprintf("*   **Model 12 Mean Deviance:** %.2f", mean(results$dev_12, na.rm=TRUE)),
  sprintf("*   **Model 17 Mean Deviance:** %.2f", mean(results$dev_17, na.rm=TRUE)),
  sprintf("*   **Paired t-test p-value:** %.4e", t.test(results$dev_12, results$dev_17, paired=TRUE)$p.value),
  "",
  "## Switch Confidence",
  sprintf("*   **Model 12:** %.2f%%", mean(results$conf_12, na.rm=TRUE)*100),
  sprintf("*   **Model 17:** %.2f%%", mean(results$conf_17, na.rm=TRUE)*100),
  "",
  "## Confusion Matrix (Model 17)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP), format_cell(results$FN)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP), format_cell(results$TN)),
  "",
  "## Fitted Exploration Parameters (Median)",
  sprintf("*   **MF Threshold:** %.4f", median(results$theta_mf, na.rm=TRUE)),
  sprintf("*   **Explore Gain:** %.4f", median(results$theta_explore, na.rm=TRUE))
)

writeLines(report, "docs/Golgi_Explore_Results.md")
cat("\nReport successfully generated in docs/Golgi_Explore_Results.md\n")
