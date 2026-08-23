library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 50
init_phi <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Golgi Variants...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi_variants.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 12 (Baseline Tanh Divisive)
  chain_tanh <- run_mcmc_subject(12, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_tanh <- eval_metrics_eccm_golgi(as.numeric(chain_tanh[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 14 (ReLU Ceiling)
  chain_ceil <- run_mcmc_subject(14, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_ceil <- eval_metrics_eccm_golgi_ceiling(as.numeric(chain_ceil[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 15 (Softmax Temperature)
  chain_soft <- run_mcmc_subject(15, iters, init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_soft <- eval_metrics_eccm_golgi_softmax(as.numeric(chain_soft[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_switch_tanh <- 0
  pred_switch_ceil <- 0
  pred_switch_soft <- 0
  actual_switches <- 0
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    
    if (!is_stay) {
      actual_switches <- actual_switches + 1
      if (ifelse(metrics_tanh$prob_ch1[i] > 0.5, 1, 2) != p_data$Resp[i-1]) pred_switch_tanh <- pred_switch_tanh + 1
      if (ifelse(metrics_ceil$prob_ch1[i] > 0.5, 1, 2) != p_data$Resp[i-1]) pred_switch_ceil <- pred_switch_ceil + 1
      if (ifelse(metrics_soft$prob_ch1[i] > 0.5, 1, 2) != p_data$Resp[i-1]) pred_switch_soft <- pred_switch_soft + 1
    }
  }
  
  data.frame(
    subject = p,
    switches = actual_switches,
    pred_switch_tanh = pred_switch_tanh,
    pred_switch_ceil = pred_switch_ceil,
    pred_switch_soft = pred_switch_soft
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

sum_switches <- sum(results$switches)
recall_tanh <- sum(results$pred_switch_tanh) / sum_switches * 100
recall_ceil <- sum(results$pred_switch_ceil) / sum_switches * 100
recall_soft <- sum(results$pred_switch_soft) / sum_switches * 100

report <- c(
  "# Golgi Variants: Pushing for +5% Switch Recall",
  "",
  "We tested two new biologically-inspired algorithms explicitly designed to maximize entropy during volatility.",
  "",
  "## Switch Recall Results",
  sprintf("*   **Model 12 (Baseline Tanh Divisive):** %.2f%%", recall_tanh),
  sprintf("*   **Model 14 (ReLU Ceiling Inhibition):** %.2f%%", recall_ceil),
  sprintf("*   **Model 15 (Temperature Softmax):** %.2f%%", recall_soft),
  "",
  "## Analysis",
  "Model 14 forces entropy by applying a dynamic Ceiling (shunting) that drops during high MF energy, capping all highly active nodes at exactly the same small value (uniformity).",
  "Model 15 explicitly guarantees max entropy by using a Softmax function where the Temperature parameter rises with MF energy, making the distribution perfectly uniform during high volatility.",
  ""
)

writeLines(report, "docs/Golgi_Variants_Results.md")
cat("\nReport successfully generated in docs/Golgi_Variants_Results.md\n")
