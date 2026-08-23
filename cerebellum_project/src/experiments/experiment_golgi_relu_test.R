library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 50
# Model 12 (Golgi Cell Divisive Normalization tanh): 9 params
init_phi_golgi <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1))

# Model 13 (Golgi Cell Subtractive Normalization ReLU): 9 params
init_phi_golgi_relu <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Golgi ReLU...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi_relu.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 12
  chain_golgi <- run_mcmc_subject(12, iters, init_phi_golgi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_golgi <- eval_metrics_eccm_golgi(as.numeric(chain_golgi[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_golgi <- eval_eccm_golgi(as.numeric(chain_golgi[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 13
  chain_relu <- run_mcmc_subject(13, iters, init_phi_golgi_relu, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_relu <- eval_metrics_eccm_golgi_relu(as.numeric(chain_relu[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_relu <- eval_eccm_golgi_relu(as.numeric(chain_relu[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_golgi <- numeric(nrow(p_data))
  pred_stay_relu <- numeric(nrow(p_data))
  
  conf_switch_golgi <- c()
  conf_switch_relu <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    pred_ch_golgi <- ifelse(metrics_golgi$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_golgi[i] <- ifelse(pred_ch_golgi == p_data$Resp[i-1], 1, 0)
    
    pred_ch_relu <- ifelse(metrics_relu$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_relu[i] <- ifelse(pred_ch_relu == p_data$Resp[i-1], 1, 0)
    
    if (!is_stay) {
      conf_switch_golgi <- c(conf_switch_golgi, ifelse(p_data$Resp[i]==1, metrics_golgi$prob_ch1[i], 1.0 - metrics_golgi$prob_ch1[i]))
      conf_switch_relu <- c(conf_switch_relu, ifelse(p_data$Resp[i]==1, metrics_relu$prob_ch1[i], 1.0 - metrics_relu$prob_ch1[i]))
    }
  }
  
  TP_G <- sum(actual_stay[-1] == 1 & pred_stay_golgi[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_G <- sum(actual_stay[-1] == 1 & pred_stay_golgi[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_G <- sum(actual_stay[-1] == 0 & pred_stay_golgi[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_G <- sum(actual_stay[-1] == 0 & pred_stay_golgi[-1] == 0) / sum(actual_stay[-1] == 0)
  
  TP_R <- sum(actual_stay[-1] == 1 & pred_stay_relu[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_R <- sum(actual_stay[-1] == 1 & pred_stay_relu[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_R <- sum(actual_stay[-1] == 0 & pred_stay_relu[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_R <- sum(actual_stay[-1] == 0 & pred_stay_relu[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    dev_golgi = dev_golgi, dev_relu = dev_relu,
    TP_G = TP_G, FN_G = FN_G, FP_G = FP_G, TN_G = TN_G,
    TP_R = TP_R, FN_R = FN_R, FP_R = FP_R, TN_R = TN_R,
    conf_golgi = mean(conf_switch_golgi, na.rm=TRUE),
    conf_relu = mean(conf_switch_relu, na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Golgi Cell Bounded ReLU: Results",
  "",
  "Comparing the **Golgi (tanh)** against the new **Golgi (ReLU)**.",
  "",
  "## 1. Deviance Check",
  sprintf("Mean Deviance Golgi (tanh): %.2f", mean(results$dev_golgi, na.rm=TRUE)),
  sprintf("Mean Deviance Golgi (ReLU): %.2f", mean(results$dev_relu, na.rm=TRUE)),
  sprintf("Paired t-test p-value: %.4e", t.test(results$dev_golgi, results$dev_relu, paired=TRUE, alternative="two.sided")$p.value),
  "",
  "## 2. Switch Confidence (Probability assigned to True Switch)",
  sprintf("Mean Switch Confidence Golgi (tanh): %.2f%%", mean(results$conf_golgi, na.rm=TRUE)*100),
  sprintf("Mean Switch Confidence Golgi (ReLU): %.2f%%", mean(results$conf_relu, na.rm=TRUE)*100),
  "",
  "## 3. Golgi Inhibition (ReLU Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_R), format_cell(results$FN_R)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_R), format_cell(results$TN_R)),
  "",
  "## 4. Golgi Inhibition (tanh Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_G), format_cell(results$FN_G)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_G), format_cell(results$TN_G))
)

writeLines(report, "docs/Golgi_ReLU_Results.md")
cat("\nReport successfully generated in docs/Golgi_ReLU_Results.md\n")
