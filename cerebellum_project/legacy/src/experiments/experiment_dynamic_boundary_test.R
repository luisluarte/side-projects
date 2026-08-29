library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 25
# Model 6 (Cortical RPE, static boundary): 8 params
init_phi_static <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

# Model 10 (Dynamic Boundary): 9 params
# a, t_nd, beta_v, eta_LTP, eta_LTD, w_cb, lambda_shift, gamma_v, gamma_a
init_phi_dynamic <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.5))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits to test Dynamic Boundary...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  sourceCpp("src/models/evaluate_metrics_dynamic_boundary.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Model 6: Standard Exponential Decay (Static Boundary)
  chain_static <- run_mcmc_subject(6, iters, init_phi_static, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_static <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_static[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_static <- eval_eccm_cortical_rpe(as.numeric(chain_static[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  # Model 10: Dynamic Boundary
  chain_dynamic <- run_mcmc_subject(10, iters, init_phi_dynamic, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_dynamic <- eval_metrics_eccm_dynamic_boundary(as.numeric(chain_dynamic[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_dynamic <- eval_eccm_dynamic_boundary(as.numeric(chain_dynamic[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  actual_stay <- numeric(nrow(p_data))
  pred_stay_static <- numeric(nrow(p_data))
  pred_stay_dynamic <- numeric(nrow(p_data))
  
  conf_static_stay <- c()
  conf_static_switch <- c()
  conf_dynamic_stay <- c()
  conf_dynamic_switch <- c()
  
  rt_static_switch <- c()
  rt_dynamic_switch <- c()
  rt_actual_switch <- c()
  
  for (i in 2:nrow(p_data)) {
    is_stay <- (p_data$Resp[i] == p_data$Resp[i-1])
    actual_stay[i] <- ifelse(is_stay, 1, 0)
    
    true_ch <- p_data$Resp[i]
    prob_true_static <- ifelse(true_ch == 1, metrics_static$prob_ch1[i], 1 - metrics_static$prob_ch1[i])
    prob_true_dynamic <- ifelse(true_ch == 1, metrics_dynamic$prob_ch1[i], 1 - metrics_dynamic$prob_ch1[i])
    
    if (is_stay) {
      conf_static_stay <- c(conf_static_stay, prob_true_static)
      conf_dynamic_stay <- c(conf_dynamic_stay, prob_true_dynamic)
    } else {
      conf_static_switch <- c(conf_static_switch, prob_true_static)
      conf_dynamic_switch <- c(conf_dynamic_switch, prob_true_dynamic)
      rt_static_switch <- c(rt_static_switch, metrics_static$exp_rt[i])
      rt_dynamic_switch <- c(rt_dynamic_switch, metrics_dynamic$exp_rt[i])
      rt_actual_switch <- c(rt_actual_switch, p_data$RT[i])
    }
    
    pred_ch_static <- ifelse(metrics_static$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_static[i] <- ifelse(pred_ch_static == p_data$Resp[i-1], 1, 0)
    
    pred_ch_dynamic <- ifelse(metrics_dynamic$prob_ch1[i] > 0.5, 1, 2)
    pred_stay_dynamic[i] <- ifelse(pred_ch_dynamic == p_data$Resp[i-1], 1, 0)
  }
  
  TP_S <- sum(actual_stay[-1] == 1 & pred_stay_static[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_S <- sum(actual_stay[-1] == 1 & pred_stay_static[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_S <- sum(actual_stay[-1] == 0 & pred_stay_static[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_S <- sum(actual_stay[-1] == 0 & pred_stay_static[-1] == 0) / sum(actual_stay[-1] == 0)
  
  TP_D <- sum(actual_stay[-1] == 1 & pred_stay_dynamic[-1] == 1) / sum(actual_stay[-1] == 1)
  FN_D <- sum(actual_stay[-1] == 1 & pred_stay_dynamic[-1] == 0) / sum(actual_stay[-1] == 1)
  FP_D <- sum(actual_stay[-1] == 0 & pred_stay_dynamic[-1] == 1) / sum(actual_stay[-1] == 0)
  TN_D <- sum(actual_stay[-1] == 0 & pred_stay_dynamic[-1] == 0) / sum(actual_stay[-1] == 0)
  
  data.frame(
    subject = p,
    dev_static = dev_static, dev_dynamic = dev_dynamic,
    TP_S = TP_S, FN_S = FN_S, FP_S = FP_S, TN_S = TN_S,
    TP_D = TP_D, FN_D = FN_D, FP_D = FP_D, TN_D = TN_D,
    conf_static_stay = mean(conf_static_stay, na.rm=TRUE),
    conf_static_switch = mean(conf_static_switch, na.rm=TRUE),
    conf_dynamic_stay = mean(conf_dynamic_stay, na.rm=TRUE),
    conf_dynamic_switch = mean(conf_dynamic_switch, na.rm=TRUE),
    rt_err_static_switch = mean(abs(rt_static_switch - rt_actual_switch), na.rm=TRUE),
    rt_err_dynamic_switch = mean(abs(rt_dynamic_switch - rt_actual_switch), na.rm=TRUE)
  )
}

stopCluster(cl)
results <- do.call(rbind, results_list)

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

report <- c(
  "# Dynamic Boundary Architecture: Results",
  "",
  "Comparing the **Static Boundary** (Model 6) against the new **Dynamic Boundary** model where Cortico-Cerebellar disagreement collapses the DDM boundary to induce impulsive panic switches.",
  "",
  "## 1. Deviance Check",
  sprintf("Mean Deviance Baseline (Static Boundary): %.2f", mean(results$dev_static, na.rm=TRUE)),
  sprintf("Mean Deviance Dynamic Boundary: %.2f", mean(results$dev_dynamic, na.rm=TRUE)),
  sprintf("Paired t-test p-value: %.4e", t.test(results$dev_static, results$dev_dynamic, paired=TRUE, alternative="two.sided")$p.value),
  "",
  "## 2. RT Prediction Error on Switches (Seconds)",
  sprintf("Mean RT Error Baseline (Static Boundary): %.4f s", mean(results$rt_err_static_switch, na.rm=TRUE)),
  sprintf("Mean RT Error Dynamic Boundary: %.4f s", mean(results$rt_err_dynamic_switch, na.rm=TRUE)),
  "",
  "## 3. Dynamic Boundary (New Model)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_D), format_cell(results$FN_D)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_D), format_cell(results$TN_D)),
  "",
  "## 4. Standard Static Boundary (Baseline)",
  "| Actual \\ Predicted | Predict Stay | Predict Switch |",
  "| :--- | :--- | :--- |",
  sprintf("| **Actual Stay** | %s | %s |", format_cell(results$TP_S), format_cell(results$FN_S)),
  sprintf("| **Actual Switch** | %s | %s |", format_cell(results$FP_S), format_cell(results$TN_S)),
  "",
  "## 5. Confidence Change (Probability assigned to True Choice)",
  "| Behavior | Static Boundary | Dynamic Boundary |",
  "| :--- | :--- | :--- |",
  sprintf("| **Stay Trials** | %s | %s |", format_cell(results$conf_static_stay), format_cell(results$conf_dynamic_stay)),
  sprintf("| **Switch Trials** | %s | %s |", format_cell(results$conf_static_switch), format_cell(results$conf_dynamic_switch))
)

writeLines(report, "docs/Dynamic_Boundary_Results.md")
cat("\nReport successfully generated in docs/Dynamic_Boundary_Results.md\n")
