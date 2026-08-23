library(Rcpp)
library(doParallel)
library(foreach)
library(glmmTMB)
library(dplyr)

cat("Starting Fast Phase 2/3 Pipeline...\n")

dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)
iters <- 25 

init_phi_0 <- c(log(2.0), log(0.3/0.7), log(3.0))
init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))
init_phi_18 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(1.0))
init_phi_19 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(0.1), log(2.0))
init_phi_20 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), 0.0, log(0.1), log(2.0))

wsls_probs <- function(resp, out) {
  prob_ch1 <- numeric(length(resp)); prob_ch1[1] <- 0.5
  for (t in 2:length(resp)) {
    if (out[t-1] == 1) { prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.999, 0.001) }
    else { prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.001, 0.999) }
  }
  return(prob_ch1)
}

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

compute_metrics <- function(probs, resp) {
  n <- length(resp)
  is_switch <- numeric(n); pred_stay <- numeric(n); p_switch <- numeric(n)
  for (i in 2:n) {
    is_switch[i] <- ifelse(resp[i] != resp[i-1], 1, 0)
    pred_ch <- ifelse(probs[i] > 0.5, 1, 2)
    pred_stay[i] <- ifelse(pred_ch == resp[i-1], 1, 0)
    if (resp[i-1] == 1) { p_switch[i] <- 1.0 - probs[i] } else { p_switch[i] <- probs[i] }
  }
  idx <- 2:n
  
  scores <- p_switch[idx]; labels <- is_switch[idx]
  ord <- order(scores, decreasing = TRUE)
  sorted_labels <- labels[ord]
  total_positives <- max(1, sum(labels == 1))
  tp <- 0; fp <- 0; precisions <- c(); recalls <- c()
  for (i in 1:length(scores)) {
    if (sorted_labels[i] == 1) { tp <- tp + 1 } else { fp <- fp + 1 }
    precisions <- c(precisions, tp / (tp + fp))
    recalls <- c(recalls, tp / total_positives)
  }
  pr_auc <- sum(diff(recalls) * (precisions[-1] + precisions[-length(precisions)]) / 2.0)
  
  max_mcc <- -1
  for (tau in seq(0.05, 0.95, by=0.05)) {
    pred <- ifelse(scores >= tau, 1, 0)
    TP_mcc <- sum(labels == 1 & pred == 1); TN_mcc <- sum(labels == 0 & pred == 0)
    FP_mcc <- sum(labels == 0 & pred == 1); FN_mcc <- sum(labels == 1 & pred == 0)
    denom <- sqrt(as.numeric(TP_mcc+FP_mcc) * as.numeric(TP_mcc+FN_mcc) * as.numeric(TN_mcc+FP_mcc) * as.numeric(TN_mcc+FN_mcc))
    mcc <- if (denom == 0) 0 else (TP_mcc*TN_mcc - FP_mcc*FN_mcc) / denom
    if (mcc > max_mcc) max_mcc <- mcc
  }
  
  actual_stay <- 1 - is_switch[idx]
  recall_stay <- sum(actual_stay == 1 & pred_stay[idx] == 1) / max(1, sum(actual_stay == 1))
  recall_switch <- sum(is_switch[idx] == 1 & pred_stay[idx] == 0) / max(1, sum(is_switch[idx] == 1))
  delta_recall <- abs(recall_stay - recall_switch)
  
  y_t <- ifelse(resp == 1, 1, 0)
  bs_t <- (probs - y_t)^2
  
  list(pr_auc=pr_auc, max_mcc=max_mcc, delta_recall=delta_recall, bs_t=bs_t, probs=probs, y_t=y_t)
}

compute_ece <- function(probs, y_t, decile_idx) {
  ece <- 0
  for (d in 1:10) {
    idx <- which(decile_idx == d)
    if (length(idx) > 0) {
      acc <- mean(y_t[idx] == ifelse(probs[idx] > 0.5, 1, 0))
      conf <- mean(ifelse(probs[idx] > 0.5, probs[idx], 1.0 - probs[idx]))
      ece <- ece + (length(idx) / length(y_t)) * abs(acc - conf)
    }
  }
  return(ece)
}

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

results_list <- foreach(p = participants, .packages = c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi_reversal.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi_asym_reversal.cpp")
  sourceCpp("src/models/evaluate_metrics_mf_rev_ablated.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) return(NULL)
  
  T <- nrow(p_data)
  rt_deciles <- as.integer(cut(p_data$RT, breaks=quantile(p_data$RT, probs=seq(0,1,0.1), na.rm=TRUE), include.lowest=TRUE))
  
  chain_6 <- run_mcmc_subject(6, iters, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  chain_18 <- run_mcmc_subject(18, iters, init_phi_18, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  chain_19 <- run_mcmc_subject(19, iters, init_phi_19, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  chain_20 <- run_mcmc_subject(20, iters, init_phi_20, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  met_6 <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_6[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  met_18 <- eval_metrics_eccm_golgi_reversal(as.numeric(chain_18[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  met_19 <- eval_metrics_eccm_golgi_asym_reversal(as.numeric(chain_19[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  met_20 <- eval_metrics_eccm_mf_rev_ablated(as.numeric(chain_20[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  probs_wsls <- wsls_probs(p_data$Resp, p_data$F)
  best_ql <- fit_qlearning(p_data$Resp, p_data$F)
  probs_ql <- numeric(T); Q <- c(0.0, 0.0)
  for (t in 1:T) {
    probs_ql[t] <- 1.0 / (1.0 + exp(-best_ql[2] * (Q[1] - Q[2])))
    ch <- p_data$Resp[t]; R <- ifelse(p_data$F[t] == 1, 1.0, 0.0)
    Q[ch] <- Q[ch] + best_ql[1] * (R - Q[ch]); unch <- ifelse(ch == 1, 2, 1)
    Q[unch] <- Q[unch] + best_ql[3] * ((1.0 - R) - Q[unch])
  }
  
  m_wsls <- compute_metrics(probs_wsls, p_data$Resp)
  m_ql <- compute_metrics(probs_ql, p_data$Resp)
  m_6 <- compute_metrics(met_6$prob_ch1, p_data$Resp)
  m_18 <- compute_metrics(met_18$prob_ch1, p_data$Resp)
  m_19 <- compute_metrics(met_19$prob_ch1, p_data$Resp)
  m_20 <- compute_metrics(met_20$prob_ch1, p_data$Resp)
  
  list(subject = p, rt_deciles=rt_deciles, m = list(WSLS=m_wsls, QL=m_ql, M6=m_6, M18=m_18, M19=m_19, M20=m_20))
}
stopCluster(cl)
results_list <- results_list[!sapply(results_list, is.null)]

models <- c("WSLS", "QL", "M6", "M18", "M19", "M20")

df_phase2 <- data.frame()
df_phase3 <- data.frame()

for (res in results_list) {
  subj <- res$subject
  for (m in models) {
    delta_r <- res$m[[m]]$delta_recall
    df_phase2 <- rbind(df_phase2, data.frame(subject=subj, model=m, delta_recall=delta_r))
    
    ece <- compute_ece(res$m[[m]]$probs, res$m[[m]]$y_t, res$rt_deciles)
    for (d in 1:10) {
      idx <- which(res$rt_deciles == d)
      if (length(idx) > 0) {
        mean_bs <- mean(res$m[[m]]$bs_t[idx])
        df_phase3 <- rbind(df_phase3, data.frame(subject=subj, model=m, decile=d, bs=mean_bs, ece=ece))
      }
    }
  }
}

df_phase2$model <- relevel(as.factor(df_phase2$model), ref="M6")
df_phase3$model <- relevel(as.factor(df_phase3$model), ref="M6")

# Correct squish transformation for strictly (0, 1) Beta bounds
squish_val <- function(x, N) (x * (N - 1) + 0.5) / N
df_phase2$delta_recall_sq <- squish_val(df_phase2$delta_recall, nrow(df_phase2))
df_phase3$bs_sq <- squish_val(df_phase3$bs, nrow(df_phase3))
df_phase3$ece_sq <- squish_val(df_phase3$ece, nrow(df_phase3))

cat("Phase 2: ZIB Regression (Delta Recall) with glmmTMB...\n")
fit_p2 <- glmmTMB(delta_recall_sq ~ model + (1 | subject), ziformula = ~ model, data = df_phase2, family = beta_family())

cat("Phase 3: Beta Regression (Brier Score across Deciles) with glmmTMB...\n")
fit_p3 <- glmmTMB(bs_sq ~ model * decile + (1 | subject), data = df_phase3, family = beta_family())

save(fit_p2, fit_p3, df_phase2, df_phase3, file="results_phase23.RData")
cat("Done.\n")
