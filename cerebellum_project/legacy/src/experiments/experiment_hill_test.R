library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 25
# Model 6 (Exponential Decay): 8 params
init_phi_exp <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

# Model 9 (Hill Equation): 9 params
# a, t_nd, beta_v, eta_LTP, eta_LTD, w_cb, lambda_shift, theta, hill_n
init_phi_hill <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(4.0 - 1.0))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Hill Equation Double Well...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  sourceCpp("src/models/evaluate_metrics_hill.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 6: Standard Exponential Decay
  chain_exp <- run_mcmc_subject(6, iters, init_phi_exp, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_exp <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_exp[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_exp <- eval_eccm_cortical_rpe(as.numeric(chain_exp[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 9: Hill Equation Double Well
  chain_hill <- run_mcmc_subject(9, iters, init_phi_hill, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_hill <- eval_metrics_eccm_hill(as.numeric(chain_hill[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_hill <- eval_eccm_hill(as.numeric(chain_hill[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_exp <- numeric(nrow(p_data))
  pred_stay_hill <- numeric(nrow(p_data))
  
  conf_exp_stay <- c()
  conf_exp_switch <- c()
  conf_hill_stay <- c()
  conf_hill_switch <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    true_ch <- p_data$Resp[i]
    prob_true_exp <- ifelse(true_ch == 1, metrics_exp$prob_ch1[i], 1 - metrics_exp$prob_ch1[i])
    prob_true_hill <- ifelse(true_ch == 1, metrics_hill$prob_ch1[i], 1 - metrics_hill$prob_ch1[i])
    
    if (is_stay) {
      conf_exp_stay <- c(conf_exp_stay, prob_true_exp)
      conf_hill_stay <- c(conf_hill_stay, prob_true_hill)
    } else {
      conf_exp_switch <- c(conf_exp_switch, prob_true_exp)
      conf_hill_switch <- c(conf_hill_switch, prob_true_hill)
    }
    
    pred_ch_exp <- ifelse(metrics_exp$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_exp[i] <- ifelse(pred_ch_exp == p_data$Resp[i-1], 1, 0)
    
    pred_ch_hill <- ifelse(metrics_hill$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_hill[i] <- ifelse(pred_ch_hill == p_data$Resp[i-1], 1, 0)
  }
  
  TP_E <- sum(actual_stay[-1] == 1 & pred_stay_exp[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_E <- sum(actual_stay[-1] == 1 & pred_stay_exp[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_E <- sum(actual_stay[-1] == 0 & pred_stay_exp[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_E <- sum(actual_stay[-1] == 0 & pred_stay_exp[-1] == 0) / sum(actual_stay[-1] == 0)
  
  TP_H <- sum(actual_stay[-1] == 1 & pred_stay_hill[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_H <- sum(actual_stay[-1] == 1 & pred_stay_hill[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_H <- sum(actual_stay[-1] == 0 & pred_stay_hill[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_H <- sum(actual_stay[-1] == 0 & pred_stay_hill[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    dev_exp = dev_exp, dev_hill = dev_hill,
    TP_E = TP_E, FN_E = FN_E, FP_E = FP_E, TN_E = TN_E,
    TP_H = TP_H, FN_H = FN_H, FP_H = FP_H, TN_H = TN_H,
    conf_exp_stay = mean(conf_exp_stay, na.rm=TRUE),
    conf_exp_switch = mean(conf_exp_switch, na.rm=TRUE),
    conf_hill_stay = mean(conf_hill_stay, na.rm=TRUE),
    conf_hill_switch = mean(conf_hill_switch, na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Hill Equation Double-Well: Results",
  "",
  "Comparing the **Exponential Decay** (Baseline) against the new **Hill Equation** double-well topology.",
  "",
  "## 1. Deviance Check",
  sprintf("Mean Deviance Baseline (Exponential): %.2f", mean(results$dev_exp, na.rm=TRUE)),
  sprintf("Mean Deviance Hill Equation (Double-Well): %.2f", mean(results$dev_hill, na.rm=TRUE)),
  sprintf("Paired t-test p-value: %.4e", t.test(results$dev_exp, results$dev_hill, paired=TRUE, alternative="two.sided")$p.value),
  "",
  "## 2. Hill Equation (New Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_H), format_cell(results$FN_H)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_H), format_cell(results$TN_H)),
  "",
  "## 3. Standard Exponential (Baseline)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_E), format_cell(results$FN_E)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_E), format_cell(results$TN_E)),
  "",
  "## 4. Confidence Change (Probability assigned to True Choice)",
  "| Behavior | Standard Exponential | Hill Equation Double-Well |",
  "| :--- | :--- | :--- |",
  sprintf("| **Stay Trials** | %s | %s |", format_cell(results$conf_exp_stay), format_cell(results$conf_hill_stay)),
  sprintf("| **Switch Trials** | %s | %s |", format_cell(results$conf_exp_switch), format_cell(results$conf_hill_switch))
)

writeLines(report, "docs/Hill_Equation_Results.md")
cat("\nReport successfully generated in docs/Hill_Equation_Results.md\n")
