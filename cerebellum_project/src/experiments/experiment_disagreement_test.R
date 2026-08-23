library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 25
init_phi_decay <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0))
init_phi_disag <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Disagreement Modulator...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_decay.cpp")
  sourceCpp("src/models/evaluate_metrics_disagreement.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Temporal Decay (Baseline)
  chain_decay <- run_mcmc_subject(3, iters, init_phi_decay, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_decay <- eval_metrics_eccm_temporal_decay(as.numeric(chain_decay[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Disagreement Modulator Only
  chain_disag <- run_mcmc_subject(5, iters, init_phi_disag, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_disag <- eval_metrics_eccm_disagreement_only(as.numeric(chain_disag[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_disag <- numeric(nrow(p_data))
  
  conf_decay_stay <- c()
  conf_decay_switch <- c()
  conf_disag_stay <- c()
  conf_disag_switch <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    true_ch <- p_data$Resp[i]
    
    # Prob assigned to TRUE choice
    prob_true_decay <- ifelse(true_ch == 1, metrics_decay$prob_ch1[i], 1 - metrics_decay$prob_ch1[i])
    prob_true_disag <- ifelse(true_ch == 1, metrics_disag$prob_ch1[i], 1 - metrics_disag$prob_ch1[i])
    
    if (is_stay) {
      conf_decay_stay <- c(conf_decay_stay, prob_true_decay)
      conf_disag_stay <- c(conf_disag_stay, prob_true_disag)
    } else {
      conf_decay_switch <- c(conf_decay_switch, prob_true_decay)
      conf_disag_switch <- c(conf_disag_switch, prob_true_disag)
    }
    
    pred_ch_disag <- ifelse(metrics_disag$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_disag[i] <- ifelse(pred_ch_disag == p_data$Resp[i-1], 1, 0)
  }
  
  # Confusion Matrix Cells (Row Normalized)
  TP_S <- sum(actual_stay[-1] == 1 & pred_stay_disag[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_S <- sum(actual_stay[-1] == 1 & pred_stay_disag[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_S <- sum(actual_stay[-1] == 0 & pred_stay_disag[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_S <- sum(actual_stay[-1] == 0 & pred_stay_disag[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    TP_S = TP_S, FN_S = FN_S, FP_S = FP_S, TN_S = TN_S,
    conf_decay_stay = mean(conf_decay_stay, na.rm=TRUE),
    conf_decay_switch = mean(conf_decay_switch, na.rm=TRUE),
    conf_disag_stay = mean(conf_disag_stay, na.rm=TRUE),
    conf_disag_switch = mean(conf_disag_switch, na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Cortico-Cerebellar Disagreement Modulator: Results",
  "",
  "Testing Option 4 (Disagreement Modulator) in isolation, compared to the Temporal Decay baseline.",
  "",
  "## 1. Confusion Matrix (Row Normalized)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_S), format_cell(results$FN_S)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_S), format_cell(results$TN_S)),
  "",
  "## 2. Confidence Change (Probability assigned to True Choice)",
  "The percentage represents the mean likelihood mass the model assigned to the correct behavior before thresholding.",
  "",
  "| Behavior | Temporal Decay (Baseline) | Disagreement Modulator |",
  "| :--- | :--- | :--- |",
  sprintf("| **Stay Trials** | %s | %s |", format_cell(results$conf_decay_stay/100), format_cell(results$conf_disag_stay/100)),
  sprintf("| **Switch Trials** | %s | %s |", format_cell(results$conf_decay_switch/100), format_cell(results$conf_disag_switch/100))
)

writeLines(report, "docs/Disagreement_Modulator_Results.md")
cat("\nReport successfully generated in docs/Disagreement_Modulator_Results.md\n")
