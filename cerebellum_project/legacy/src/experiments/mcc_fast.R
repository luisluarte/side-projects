library(Rcpp)
library(dplyr)
sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
sourceCpp("src/models/evaluate_metrics_golgi_reversal.cpp")
sourceCpp("src/models/evaluate_metrics_golgi_asym_reversal.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))
init_phi_19 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(0.1), log(2.0))

wsls_probs <- function(resp, out) {
  prob_ch1 <- numeric(length(resp)); prob_ch1[1] <- 0.5
  for (t in 2:length(resp)) {
    if (out[t-1] == 1) { prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.999, 0.001) }
    else { prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.001, 0.999) }
  }
  return(prob_ch1)
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

mcc_6 <- c(); mcc_19 <- c(); mcc_wsls <- c()

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) next
  
  chain_6 <- run_mcmc_subject(6, 10, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  chain_19 <- run_mcmc_subject(19, 10, init_phi_19, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  met_6 <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_6[10, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  met_19 <- eval_metrics_eccm_golgi_asym_reversal(as.numeric(chain_19[10, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  probs_wsls <- wsls_probs(p_data$Resp, p_data$F)
  
  mcc_6 <- c(mcc_6, compute_mcc(met_6$prob_ch1, p_data$Resp))
  mcc_19 <- c(mcc_19, compute_mcc(met_19$prob_ch1, p_data$Resp))
  mcc_wsls <- c(mcc_wsls, compute_mcc(probs_wsls, p_data$Resp))
}

cat(sprintf("WSLS Mean Max-MCC: %.4f\n", mean(mcc_wsls)))
cat(sprintf("Baseline ECCM (M6) Mean Max-MCC: %.4f\n", mean(mcc_6)))
cat(sprintf("Asymmetric Reversal (M19) Mean Max-MCC: %.4f\n", mean(mcc_19)))
