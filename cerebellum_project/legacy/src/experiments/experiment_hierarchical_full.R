library(Rcpp)
library(doParallel)
library(foreach)
library(loo)
library(brms)
library(transport)
library(RWiener)
library(dplyr)

cat("Starting Hierarchical Model Evaluation Pipeline...\n")

dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)
iters <- 75 

init_phi_0 <- c(log(2.0), log(0.3/0.7), log(3.0))
init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))
init_phi_18 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(1.0))
init_phi_19 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(0.1), log(2.0))
init_phi_20 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), 0.0, log(0.1), log(2.0))

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
          Q[ch] <- Q[ch] + alpha * (R - Q[ch])
          unch <- ifelse(ch == 1, 2, 1)
          Q[unch] <- Q[unch] + alpha_c * ((1.0 - R) - Q[unch])
        }
        if (nll < best_nll) { best_nll <- nll; best_params <- c(alpha, beta_val, alpha_c, 0.0) }
      }
    }
  }
  return(best_params)
}

wsls_probs <- function(resp, out) {
  prob_ch1 <- numeric(length(resp)); prob_ch1[1] <- 0.5
  for (t in 2:length(resp)) {
    if (out[t-1] == 1) { prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.999, 0.001) }
    else { prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.001, 0.999) }
  }
  return(prob_ch1)
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
  
  # PR-AUC
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
  
  # Max-MCC
  max_mcc <- -1
  for (tau in seq(0.05, 0.95, by=0.05)) {
    pred <- ifelse(scores >= tau, 1, 0)
    TP_mcc <- sum(labels == 1 & pred == 1)
    TN_mcc <- sum(labels == 0 & pred == 0)
    FP_mcc <- sum(labels == 0 & pred == 1)
    FN_mcc <- sum(labels == 1 & pred == 0)
    denom <- sqrt(as.numeric(TP_mcc+FP_mcc) * as.numeric(TP_mcc+FN_mcc) * as.numeric(TN_mcc+FP_mcc) * as.numeric(TN_mcc+FN_mcc))
    mcc <- if (denom == 0) 0 else (TP_mcc*TN_mcc - FP_mcc*FN_mcc) / denom
    if (mcc > max_mcc) max_mcc <- mcc
  }
  
  # Delta Recall
  actual_stay <- 1 - is_switch[idx]
  recall_stay <- sum(actual_stay == 1 & pred_stay[idx] == 1) / max(1, sum(actual_stay == 1))
  recall_switch <- sum(is_switch[idx] == 1 & pred_stay[idx] == 0) / max(1, sum(is_switch[idx] == 1))
  delta_recall <- abs(recall_stay - recall_switch)
  
  # Brier Score (trial by trial)
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

cat("Running MCMC and computing metrics per subject...\n")
results_list <- foreach(p = participants, .packages = c("Rcpp", "RWiener", "transport")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/fitting_procedures/extract_pointwise_ll.cpp")
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
  
  ll_6 <- extract_all_pointwise_ll(6, chain_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_18 <- extract_all_pointwise_ll(18, chain_18, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_19 <- extract_all_pointwise_ll(19, chain_19, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_20 <- extract_all_pointwise_ll(20, chain_20, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
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
  
  # RT simulation for Phase 4 (simplified heuristic to get Wiener RTs from current drift)
  # W1 distance per decile
  get_w1 <- function(phi) {
      a <- exp(phi[1]); t_nd <- 1.0 / (1.0 + exp(-phi[2])); 
      # Simplification: use mean simulated RTs vs empirical
      w1_dec <- numeric(10)
      for (d in 1:10) {
          emp <- p_data$RT[rt_deciles == d]
          if(length(emp) > 0) {
              sim <- rwiener(length(emp), alpha=a, tau=t_nd, beta=0.5, delta=0.5)$q # rough proxy
              w1_dec[d] <- wasserstein1d(emp, sim)
          } else { w1_dec[d] <- NA }
      }
      return(w1_dec)
  }
  
  w1_wsls <- rep(NA, 10) # WSLS has no RT
  w1_ql <- rep(NA, 10)
  w1_6 <- get_w1(chain_6[iters, ])
  w1_18 <- get_w1(chain_18[iters, ])
  w1_19 <- get_w1(chain_19[iters, ])
  w1_20 <- get_w1(chain_20[iters, ])
  
  list(subject = p, rt_deciles=rt_deciles,
       ll = list(M6=ll_6, M18=ll_18, M19=ll_19, M20=ll_20),
       m = list(WSLS=m_wsls, QL=m_ql, M6=m_6, M18=m_18, M19=m_19, M20=m_20),
       w1 = list(WSLS=w1_wsls, QL=w1_ql, M6=w1_6, M18=w1_18, M19=w1_19, M20=w1_20))
}
stopCluster(cl)
results_list <- results_list[!sapply(results_list, is.null)]

models <- c("WSLS", "QL", "M6", "M18", "M19", "M20")

# Phase 1: PSIS-LOO
cat("Phase 1: PSIS-LOO...\n")
elpd_diffs <- numeric(length(models))
names(elpd_diffs) <- models
base_elpd <- 0
for (m in c("M6", "M18", "M19", "M20")) {
  loo_val <- 0
  for (res in results_list) {
    mat <- res$ll[[m]]
    mat[is.na(mat) | is.infinite(mat)] <- -1e9
    loo_obj <- suppressWarnings(loo(mat))
    loo_val <- loo_val + loo_obj$estimates["elpd_loo", "Estimate"]
  }
  if (m == "M6") base_elpd <- loo_val
  elpd_diffs[m] <- loo_val - base_elpd
}

# Construct DataFrames for BRMS
df_phase2 <- data.frame()
df_phase3 <- data.frame()
df_phase4 <- data.frame()

squish <- function(x) (x * (length(x) - 1) + 0.5) / length(x)

for (res in results_list) {
  subj <- res$subject
  for (m in models) {
    # Phase 2
    delta_r <- res$m[[m]]$delta_recall
    df_phase2 <- rbind(df_phase2, data.frame(subject=subj, model=m, delta_recall=squish(delta_r)))
    
    # Phase 3
    ece <- compute_ece(res$m[[m]]$probs, res$m[[m]]$y_t, res$rt_deciles)
    for (d in 1:10) {
      idx <- which(res$rt_deciles == d)
      if (length(idx) > 0) {
        mean_bs <- mean(res$m[[m]]$bs_t[idx])
        df_phase3 <- rbind(df_phase3, data.frame(subject=subj, model=m, decile=d, bs=squish(mean_bs), ece=squish(ece)))
        
        # Phase 4
        w1_val <- res$w1[[m]][d]
        if (!is.na(w1_val)) {
            df_phase4 <- rbind(df_phase4, data.frame(subject=subj, model=m, decile=d, w1=w1_val))
        }
      }
    }
  }
}

df_phase2$model <- as.factor(df_phase2$model)
df_phase3$model <- as.factor(df_phase3$model)
df_phase4$model <- as.factor(df_phase4$model)

cat("Phase 2: ZIB Regression (Delta Recall)...\n")
fit_p2 <- brm(bf(delta_recall ~ model + (1 | subject), zi ~ model), data = df_phase2, family = zero_inflated_beta(), chains = 2, iter = 1000, backend = "cmdstanr", silent = 2, refresh=0)

cat("Phase 3: Beta Regression (Brier Score across Deciles)...\n")
fit_p3 <- brm(bs ~ model * decile + (1 | subject), data = df_phase3, family = Beta(), chains = 2, iter = 1000, backend = "cmdstanr", silent = 2, refresh=0)

cat("Phase 4: Gamma Log-Link Regression (W1 Distance across Deciles)...\n")
df_phase4 <- df_phase4[df_phase4$model %in% c("M6", "M18", "M19", "M20"), ]
fit_p4 <- brm(w1 ~ model * decile + (1 | subject), data = df_phase4, family = Gamma(link="log"), chains = 2, iter = 1000, backend = "cmdstanr", silent = 2, refresh=0)

cat("Generating Report...\n")
sink("docs/hierarchical_results.md")
cat("# Hierarchical Evaluation Results\n\n")
cat("## Phase 1: PSIS-LOO (Out-of-Sample Supremacy)\n")
for (m in c("M6", "M18", "M19", "M20")) {
  cat(sprintf("- **%s**: Delta elpd = %.2f\n", m, elpd_diffs[m]))
}
cat("\n## Phase 2: ZIB Regression (Delta Recall)\n```\n")
print(summary(fit_p2))
cat("```\n\n## Phase 3: Beta Regression (Epistemic Calibration)\n```\n")
print(summary(fit_p3))
cat("```\n\n## Phase 4: Gamma Regression (Kinetic Alignment W1)\n```\n")
print(summary(fit_p4))
cat("```\n")
sink()

cat("Pipeline finished. Results in docs/hierarchical_results.md\n")
