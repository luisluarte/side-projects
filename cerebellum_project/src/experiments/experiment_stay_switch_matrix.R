library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 25
init_phi_intact <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)
init_phi_decay  <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to build Stay/Switch Confusion Matrix...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics.cpp")
  sourceCpp("src/models/evaluate_metrics_decay.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  chain_intact <- run_mcmc_subject(1, iters, init_phi_intact, p_data$Resp, p_data$F, p_data$RT)
  metrics_intact <- eval_metrics_eccm(as.numeric(chain_intact[iters, ]), p_data$Resp, p_data$F, p_data$RT, FALSE)
  
  chain_decay <- run_mcmc_subject(3, iters, init_phi_decay, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_decay <- eval_metrics_eccm_temporal_decay(as.numeric(chain_decay[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_intact <- numeric(nrow(p_data))
  pred_stay_decay <- numeric(nrow(p_data))
  
  for (i in 2:nrow(p_data)) {
    actual_stay[i] <- ifelse(p_data$Resp[i] == p_data$Resp[i-1], 1, 0)
    
    pred_ch_intact <- ifelse(metrics_intact$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_intact[i] <- ifelse(pred_ch_intact == p_data$Resp[i-1], 1, 0)
    
    pred_ch_decay <- ifelse(metrics_decay$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_decay[i] <- ifelse(pred_ch_decay == p_data$Resp[i-1], 1, 0)
  }
  
  # Confusion Matrix Cells (Percentages per subject)
  N <- nrow(p_data) - 1
  
  # Intact (Row Normalized)
  TP_I <- sum(actual_stay[-1] == 1 & pred_stay_intact[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_I <- sum(actual_stay[-1] == 1 & pred_stay_intact[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_I <- sum(actual_stay[-1] == 0 & pred_stay_intact[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_I <- sum(actual_stay[-1] == 0 & pred_stay_intact[-1] == 0) / sum(actual_stay[-1] == 0)
  
  # Decay (Row Normalized)
  TP_D <- sum(actual_stay[-1] == 1 & pred_stay_decay[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_D <- sum(actual_stay[-1] == 1 & pred_stay_decay[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_D <- sum(actual_stay[-1] == 0 & pred_stay_decay[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_D <- sum(actual_stay[-1] == 0 & pred_stay_decay[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    TP_I = TP_I, TN_I = TN_I, FP_I = FP_I, FN_I = FN_I,
    TP_D = TP_D, TN_D = TN_D, FP_D = FP_D, FN_D = FN_D
  )
}

stopCluster(cl)

results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec)*100, sd(vec)*100)
}

report <- c(
  "# Stay/Switch Confusion Matrices (Means ± SD)",
  "",
  "The matrices below represent the percentage of trials where the model correctly predicted whether the participant would **Stay** (repeat the same choice) or **Switch** (change choice) compared to actual behavior.",
  "",
  "## Temporal Decay ECCM",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_D), format_cell(results$FN_D)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_D), format_cell(results$TN_D)),
  "",
  "## Original Intact ECCM",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_I), format_cell(results$FN_I)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_I), format_cell(results$TN_I)),
  "",
  "## Analysis",
  "The matrices highlight how effectively the Temporal Decay model captures both stationary behavior (Stay) and volatile behavior (Switch)."
)

writeLines(report, "docs/Stay_Switch_Confusion_Matrix.md")
cat("\nReport successfully generated in docs/Stay_Switch_Confusion_Matrix.md\n")
