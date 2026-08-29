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

cat("Running parallel MCMC fits to test Cortical RPE Feedback...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_switch.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 4: Switch Mechanisms (Cerebellar Error)
  chain_switch <- run_mcmc_subject(4, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_switch <- eval_metrics_eccm_switch_mechanisms(as.numeric(chain_switch[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_switch <- eval_eccm_switch_mechanisms(as.numeric(chain_switch[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 6: Cortical RPE Feedback
  chain_cortical <- run_mcmc_subject(6, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_cortical <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_cortical[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_cortical <- eval_eccm_cortical_rpe(as.numeric(chain_cortical[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_switch <- numeric(nrow(p_data))
  pred_stay_cortical <- numeric(nrow(p_data))
  
  conf_switch_stay <- c()
  conf_switch_switch <- c()
  conf_cortical_stay <- c()
  conf_cortical_switch <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    true_ch <- p_data$Resp[i]
    prob_true_switch <- ifelse(true_ch == 1, metrics_switch$prob_ch1[i], 1 - metrics_switch$prob_ch1[i])
    prob_true_cortical <- ifelse(true_ch == 1, metrics_cortical$prob_ch1[i], 1 - metrics_cortical$prob_ch1[i])
    
    if (is_stay) {
      conf_switch_stay <- c(conf_switch_stay, prob_true_switch)
      conf_cortical_stay <- c(conf_cortical_stay, prob_true_cortical)
    } else {
      conf_switch_switch <- c(conf_switch_switch, prob_true_switch)
      conf_cortical_switch <- c(conf_cortical_switch, prob_true_cortical)
    }
    
    pred_ch_switch <- ifelse(metrics_switch$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_switch[i] <- ifelse(pred_ch_switch == p_data$Resp[i-1], 1, 0)
    
    pred_ch_cortical <- ifelse(metrics_cortical$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_cortical[i] <- ifelse(pred_ch_cortical == p_data$Resp[i-1], 1, 0)
  }
  
  TP_S <- sum(actual_stay[-1] == 1 & pred_stay_switch[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_S <- sum(actual_stay[-1] == 1 & pred_stay_switch[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_S <- sum(actual_stay[-1] == 0 & pred_stay_switch[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_S <- sum(actual_stay[-1] == 0 & pred_stay_switch[-1] == 0) / sum(actual_stay[-1] == 0)
  
  TP_C <- sum(actual_stay[-1] == 1 & pred_stay_cortical[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_C <- sum(actual_stay[-1] == 1 & pred_stay_cortical[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_C <- sum(actual_stay[-1] == 0 & pred_stay_cortical[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_C <- sum(actual_stay[-1] == 0 & pred_stay_cortical[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    dev_switch = dev_switch, dev_cortical = dev_cortical,
    TP_S = TP_S, FN_S = FN_S, FP_S = FP_S, TN_S = TN_S,
    TP_C = TP_C, FN_C = FN_C, FP_C = FP_C, TN_C = TN_C,
    conf_switch_stay = mean(conf_switch_stay, na.rm=TRUE),
    conf_switch_switch = mean(conf_switch_switch, na.rm=TRUE),
    conf_cortical_stay = mean(conf_cortical_stay, na.rm=TRUE),
    conf_cortical_switch = mean(conf_cortical_switch, na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Cortical RPE Feedback: Results",
  "",
  "Comparing **Option 2** (Cerebellar Error in Mossy Fibers) against the new **Cortical RPE Feedback** (Cortical Error in Mossy Fibers). Both models use the 240-D shift register and Disagreement Modulation.",
  "",
  "## 1. Deviance Check",
  sprintf("Mean Deviance Cerebellar Error: %.2f", mean(results$dev_switch, na.rm=TRUE)),
  sprintf("Mean Deviance Cortical RPE: %.2f", mean(results$dev_cortical, na.rm=TRUE)),
  sprintf("Paired t-test p-value: %.4e", t.test(results$dev_switch, results$dev_cortical, paired=TRUE, alternative="greater")$p.value),
  "",
  "## 2. Cortical RPE Feedback (New Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_C), format_cell(results$FN_C)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_C), format_cell(results$TN_C)),
  "",
  "## 3. Cerebellar Error Feedback (Previous Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_S), format_cell(results$FN_S)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_S), format_cell(results$TN_S)),
  "",
  "## 4. Confidence Change (Probability assigned to True Choice)",
  "| Behavior | Cerebellar Error | Cortical RPE |",
  "| :--- | :--- | :--- |",
  sprintf("| **Stay Trials** | %s | %s |", format_cell(results$conf_switch_stay), format_cell(results$conf_cortical_stay)),
  sprintf("| **Switch Trials** | %s | %s |", format_cell(results$conf_switch_switch), format_cell(results$conf_cortical_switch))
)

writeLines(report, "docs/Cortical_RPE_Results.md")
cat("\nReport successfully generated in docs/Cortical_RPE_Results.md\n")
