library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 25
init_phi_decay  <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0))
init_phi_switch <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Switch Mechanisms...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_decay.cpp")
  sourceCpp("src/models/evaluate_metrics_switch.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Temporal Decay (Model 3)
  chain_decay <- run_mcmc_subject(3, iters, init_phi_decay, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_decay <- eval_metrics_eccm_temporal_decay(as.numeric(chain_decay[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_decay <- eval_eccm_temporal_decay(as.numeric(chain_decay[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Switch Mechanisms (Model 4)
  chain_switch <- run_mcmc_subject(4, iters, init_phi_switch, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_switch <- eval_metrics_eccm_switch_mechanisms(as.numeric(chain_switch[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_switch <- eval_eccm_switch_mechanisms(as.numeric(chain_switch[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_decay <- numeric(nrow(p_data))
  pred_stay_switch <- numeric(nrow(p_data))
  
  for (i in 2:nrow(p_data)) {
    actual_stay[i] <- ifelse(p_data$Resp[i] == p_data$Resp[i-1], 1, 0)
    
    pred_ch_decay <- ifelse(metrics_decay$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_decay[i] <- ifelse(pred_ch_decay == p_data$Resp[i-1], 1, 0)
    
    pred_ch_switch <- ifelse(metrics_switch$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_switch[i] <- ifelse(pred_ch_switch == p_data$Resp[i-1], 1, 0)
  }
  
  # Confusion Matrix Cells (Row Normalized)
  TP_D <- sum(actual_stay[-1] == 1 & pred_stay_decay[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_D <- sum(actual_stay[-1] == 1 & pred_stay_decay[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_D <- sum(actual_stay[-1] == 0 & pred_stay_decay[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_D <- sum(actual_stay[-1] == 0 & pred_stay_decay[-1] == 0) / sum(actual_stay[-1] == 0)
  
  TP_S <- sum(actual_stay[-1] == 1 & pred_stay_switch[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_S <- sum(actual_stay[-1] == 1 & pred_stay_switch[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_S <- sum(actual_stay[-1] == 0 & pred_stay_switch[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_S <- sum(actual_stay[-1] == 0 & pred_stay_switch[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    dev_decay = dev_decay, dev_switch = dev_switch,
    TP_D = TP_D, FN_D = FN_D, FP_D = FP_D, TN_D = TN_D,
    TP_S = TP_S, FN_S = FN_S, FP_S = FP_S, TN_S = TN_S
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Switch Prediction Mechanisms: Results",
  "",
  "Comparing the baseline **Temporal Decay Model** (7 params) against the new **Combined Switch Mechanisms Model** (8 params).",
  "",
  "## 1. Deviance Check (Is the 8th parameter justified?)",
  sprintf("Mean Deviance Decay: %.2f", mean(results$dev_decay, na.rm=TRUE)),
  sprintf("Mean Deviance Switch: %.2f", mean(results$dev_switch, na.rm=TRUE)),
  sprintf("Paired t-test p-value: %.4e", t.test(results$dev_decay, results$dev_switch, paired=TRUE, alternative="greater")$p.value),
  "",
  "## 2. Combined Switch Mechanisms (New Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_S), format_cell(results$FN_S)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_S), format_cell(results$TN_S)),
  "",
  "## 3. Temporal Decay (Baseline)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_D), format_cell(results$FN_D)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_D), format_cell(results$TN_D))
)

writeLines(report, "docs/Switch_Mechanisms_Results.md")
cat("\nReport successfully generated in docs/Switch_Mechanisms_Results.md\n")
