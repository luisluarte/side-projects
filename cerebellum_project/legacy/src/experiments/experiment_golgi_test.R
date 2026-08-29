library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 50
# Model 6 (Cortical RPE, baseline): 8 params
init_phi_baseline <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

# Model 12 (Golgi Cell Divisive Normalization): 9 params
# a, t_nd, beta_v, eta_LTP, eta_LTD, w_cb, lambda_shift, gamma_suppress, theta_golgi
init_phi_golgi <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Golgi Inhibition...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 6
  chain_base <- run_mcmc_subject(6, iters, init_phi_baseline, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_base <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_base[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_base <- eval_eccm_cortical_rpe(as.numeric(chain_base[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 12
  chain_golgi <- run_mcmc_subject(12, iters, init_phi_golgi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_golgi <- eval_metrics_eccm_golgi(as.numeric(chain_golgi[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_golgi <- eval_eccm_golgi(as.numeric(chain_golgi[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_base <- numeric(nrow(p_data))
  pred_stay_golgi <- numeric(nrow(p_data))
  
  conf_switch_base <- c()
  conf_switch_golgi <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    pred_ch_base <- ifelse(metrics_base$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_base[i] <- ifelse(pred_ch_base == p_data$Resp[i-1], 1, 0)
    
    pred_ch_golgi <- ifelse(metrics_golgi$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_golgi[i] <- ifelse(pred_ch_golgi == p_data$Resp[i-1], 1, 0)
    
    if (!is_stay) {
      conf_switch_base <- c(conf_switch_base, ifelse(p_data$Resp[i]==1, metrics_base$prob_ch1[i], 1.0 - metrics_base$prob_ch1[i]))
      conf_switch_golgi <- c(conf_switch_golgi, ifelse(p_data$Resp[i]==1, metrics_golgi$prob_ch1[i], 1.0 - metrics_golgi$prob_ch1[i]))
    }
  }
  
  TP_B <- sum(actual_stay[-1] == 1 & pred_stay_base[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_B <- sum(actual_stay[-1] == 1 & pred_stay_base[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_B <- sum(actual_stay[-1] == 0 & pred_stay_base[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_B <- sum(actual_stay[-1] == 0 & pred_stay_base[-1] == 0) / sum(actual_stay[-1] == 0)
  
  TP_G <- sum(actual_stay[-1] == 1 & pred_stay_golgi[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_G <- sum(actual_stay[-1] == 1 & pred_stay_golgi[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_G <- sum(actual_stay[-1] == 0 & pred_stay_golgi[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_G <- sum(actual_stay[-1] == 0 & pred_stay_golgi[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    dev_base = dev_base, dev_golgi = dev_golgi,
    TP_B = TP_B, FN_B = FN_B, FP_B = FP_B, TN_B = TN_B,
    TP_G = TP_G, FN_G = FN_G, FP_G = FP_G, TN_G = TN_G,
    conf_base = mean(conf_switch_base, na.rm=TRUE),
    conf_golgi = mean(conf_switch_golgi, na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Golgi Cell Divisive Normalization: Results",
  "",
  "Comparing the **Baseline Cortical RPE** against the new **Golgi Inhibition Model**.",
  "",
  "## 1. Deviance Check",
  sprintf("Mean Deviance Baseline: %.2f", mean(results$dev_base, na.rm=TRUE)),
  sprintf("Mean Deviance Golgi Model: %.2f", mean(results$dev_golgi, na.rm=TRUE)),
  sprintf("Paired t-test p-value: %.4e", t.test(results$dev_base, results$dev_golgi, paired=TRUE, alternative="two.sided")$p.value),
  "",
  "## 2. Switch Confidence (Probability assigned to True Switch)",
  sprintf("Mean Switch Confidence Baseline: %.2f%%", mean(results$conf_base, na.rm=TRUE)*100),
  sprintf("Mean Switch Confidence Golgi Model: %.2f%%", mean(results$conf_golgi, na.rm=TRUE)*100),
  "",
  "## 3. Golgi Inhibition (New Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_G), format_cell(results$FN_G)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_G), format_cell(results$TN_G)),
  "",
  "## 4. Baseline Model",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_B), format_cell(results$FN_B)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_B), format_cell(results$TN_B))
)

writeLines(report, "docs/Golgi_Inhibition_Results.md")
cat("\nReport successfully generated in docs/Golgi_Inhibition_Results.md\n")
