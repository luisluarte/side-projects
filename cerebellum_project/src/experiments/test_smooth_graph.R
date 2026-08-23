library(Rcpp)
library(pROC)
sourceCpp("src/models/eccm_smooth_graph.cpp")
sourceCpp("src/models/evaluate_metrics_smooth_graph.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

# phi: a, t_nd, beta_v, eta_win, eta_loss, w_cb, lambda_shift, gamma_suppress, alpha_diff
init_phi_smooth <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.1/0.9), log(0.88/0.12), 0.5, log(1.0), log(0.5), log(2.0))

ll_smooth <- 0
all_probs <- c()
all_actual <- c()

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) next
  
  met <- eval_metrics_smooth_graph(init_phi_smooth, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_smooth <- ll_smooth - eval_eccm_smooth_graph(init_phi_smooth, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  all_probs <- c(all_probs, met$prob_ch1)
  actual <- ifelse(p_data$Resp == 1, 1, 0)
  all_actual <- c(all_actual, actual)
}

sink("output_smooth.txt")
cat(sprintf("Smooth Graph (DDM) pseudo-LOO sum: %.2f\n", ll_smooth))
roc_obj <- roc(all_actual, all_probs, quiet=TRUE)
auc_val <- auc(roc_obj)
cat(sprintf("Smooth Graph ROC-AUC: %.4f\n", auc_val))
sink()
