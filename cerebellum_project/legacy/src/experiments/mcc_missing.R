library(Rcpp)
library(dplyr)
sourceCpp("src/models/evaluate_metrics_golgi_reversal.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

init_phi_18 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(1.0))

fit_qlearning <- function(resp, out) {
  best_nll <- Inf; best_params <- c(0.1, 1.0, 0.1, 0.0)
  for (alpha in c(0.05, 0.2, 0.5)) {
    for (beta_val in c(1.0, 3.0)) {
      for (alpha_c in c(0.0, 0.1)) {
        Q <- c(0.0, 0.0); nll <- 0
        for (t in 1:length(resp)) {
          p1 <- 1.0 / (1.0 + exp(-beta_val * (Q[1] - Q[2])))
          p <- ifelse(resp[t] == 1, p1, 1.0 - p1)
          nll <- nll - log(max(p, 1e-8))
          ch <- resp[t]; R <- ifelse(out[t] == 1, 1.0, 0.0)
          Q[ch] <- Q[ch] + alpha * (R - Q[ch]); unch <- ifelse(ch == 1, 2, 1)
          Q[unch] <- Q[unch] + alpha_c * ((1.0 - R) - Q[unch])
        }
        if (nll < best_nll) { best_nll <- nll; best_params <- c(alpha, beta_val, alpha_c, 0.0) }
      }
    }
  }
  return(best_params)
}

compute_mcc <- function(probs, resp) {
  n <- length(resp)
  is_switch <- numeric(n); p_switch <- numeric(n)
  for (i in 2:n) {
    is_switch[i] <- ifelse(resp[i] != resp[i-1], 1, 0)
    if (resp[i-1] == 1) { p_switch[i] <- 1.0 - probs[i] } else { p_switch[i] <- probs[i] }
  }
  idx <- 2:n
  scores <- p_switch[idx]; labels <- is_switch[idx]
  
  max_mcc <- -1
  for (tau in seq(0.05, 0.95, by=0.05)) {
    pred <- ifelse(scores >= tau, 1, 0)
    TP <- sum(labels == 1 & pred == 1); TN <- sum(labels == 0 & pred == 0)
    FP <- sum(labels == 0 & pred == 1); FN <- sum(labels == 1 & pred == 0)
    denom <- sqrt(as.numeric(TP+FP) * as.numeric(TP+FN) * as.numeric(TN+FP) * as.numeric(TN+FN))
    mcc <- if (denom == 0) 0 else (TP*TN - FP*FN) / denom
    if (mcc > max_mcc) max_mcc <- mcc
  }
  return(max_mcc)
}

mcc_18 <- c(); mcc_ql <- c()

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) next
  
  met_18 <- eval_metrics_eccm_golgi_reversal(init_phi_18, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  best_ql <- fit_qlearning(p_data$Resp, p_data$F)
  probs_ql <- numeric(nrow(p_data)); Q <- c(0.0, 0.0)
  for (t in 1:nrow(p_data)) {
    probs_ql[t] <- 1.0 / (1.0 + exp(-best_ql[2] * (Q[1] - Q[2])))
    ch <- p_data$Resp[t]; R <- ifelse(p_data$F[t] == 1, 1.0, 0.0)
    Q[ch] <- Q[ch] + best_ql[1] * (R - Q[ch]); unch <- ifelse(ch == 1, 2, 1)
    Q[unch] <- Q[unch] + best_ql[3] * ((1.0 - R) - Q[unch])
  }
  
  mcc_18 <- c(mcc_18, compute_mcc(met_18$prob_ch1, p_data$Resp))
  mcc_ql <- c(mcc_ql, compute_mcc(probs_ql, p_data$Resp))
}

cat(sprintf("Q-Learning Mean Max-MCC: %.4f\n", mean(mcc_ql)))
cat(sprintf("Symmetric Reversal (M18) Mean Max-MCC: %.4f\n", mean(mcc_18)))
