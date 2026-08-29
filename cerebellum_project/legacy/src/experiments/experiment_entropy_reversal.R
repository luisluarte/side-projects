library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 50
# Model 16 has 10 parameters. The 10th parameter is theta_rev (log-transformed)
init_phi <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), log(2.0))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Model 16: Entropy-Driven Reversal...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi.cpp")
  sourceCpp("src/models/evaluate_metrics_entropy_reversal.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 12 (Baseline Tanh Divisive - 9 params)
  init_phi_12 <- init_phi[1:9]
  chain_tanh <- run_mcmc_subject(12, iters, init_phi_12, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_tanh <- eval_metrics_eccm_golgi(as.numeric(chain_tanh[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 16 (Entropy Reversal - 10 params)
  chain_rev <- run_mcmc_subject(16, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_rev <- eval_metrics_eccm_entropy_reversal(as.numeric(chain_rev[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_rev <- eval_eccm_entropy_reversal(as.numeric(chain_rev[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_rev <- numeric(nrow(p_data))
  
  pred_switch_tanh <- 0
  pred_switch_rev <- 0
  actual_switches <- 0
  conf_switch_rev <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    pred_ch_rev <- ifelse(metrics_rev$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_rev[i] <- ifelse(pred_ch_rev == p_data$Resp[i-1], 1, 0)
    
    if (!is_stay) {
      actual_switches <- actual_switches + 1
      if (ifelse(metrics_tanh$prob_ch1[i] > 0.5, 1, 2) != p_data$Resp[i-1]) pred_switch_tanh <- pred_switch_tanh + 1
      if (ifelse(metrics_rev$prob_ch1[i] > 0.5, 1, 2) != p_data$Resp[i-1]) pred_switch_rev <- pred_switch_rev + 1
      
      conf_switch_rev <- c(conf_switch_rev, ifelse(p_data$Resp[i]==1, metrics_rev$prob_ch1[i], 1.0 - metrics_rev$prob_ch1[i]))
    }
  }
  
  TP_R <- sum(actual_stay[-1] == 1 & pred_stay_rev[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_R <- sum(actual_stay[-1] == 1 & pred_stay_rev[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_R <- sum(actual_stay[-1] == 0 & pred_stay_rev[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_R <- sum(actual_stay[-1] == 0 & pred_stay_rev[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    switches = actual_switches,
    pred_switch_tanh = pred_switch_tanh,
    pred_switch_rev = pred_switch_rev,
    dev_rev = dev_rev,
    TP_R = TP_R, FN_R = FN_R, FP_R = FP_R, TN_R = TN_R,
    conf_rev = mean(conf_switch_rev, na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

sum_switches <- sum(results$switches)
recall_tanh <- sum(results$pred_switch_tanh) / sum_switches * 100
recall_rev <- sum(results$pred_switch_rev) / sum_switches * 100

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# ECCM Model 16: Entropy-Driven Reversal",
  "",
  "We tested whether directly mapping GC Layer Entropy to a drift-inverting exploration term forces actual switches.",
  "",
  "## Switch Recall Results",
  sprintf("*   **Model 12 (Baseline Tanh Divisive):** %.2f%%", recall_tanh),
  sprintf("*   **Model 16 (Entropy Reversal):** %.2f%%", recall_rev),
  "",
  "## Confusion Matrix (Model 16)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_R), format_cell(results$FN_R)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_R), format_cell(results$TN_R)),
  "",
  "## Confidence & Deviance",
  sprintf("*   **Mean Switch Confidence:** %.2f%%", mean(results$conf_rev, na.rm=TRUE)*100),
  sprintf("*   **Mean Deviance:** %.2f", mean(results$dev_rev, na.rm=TRUE))
)

writeLines(report, "docs/Entropy_Reversal_Results.md")
cat("\nReport successfully generated in docs/Entropy_Reversal_Results.md\n")
