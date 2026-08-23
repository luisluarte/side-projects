library(Rcpp)

setwd(this.path::here())
sourceCpp("../../src/fitting_procedures/mcmc_sampler.cpp")
sourceCpp("../../src/models/evaluate_metrics_bvk.cpp")
sourceCpp("../../src/models/evaluate_metrics_cortical_rpe.cpp")

dat_all <- read.csv("../../data/raw/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 75

# BVK (Model 23): 10 params
init_phi_23 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), 0.0, log(0.5))
# Baseline ECCM (Model 6): 8 params
init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

# ============================================================================
# PR-AUC computation
# ============================================================================
compute_pr_auc <- function(scores, labels) {
  n <- length(scores)
  ord <- order(scores, decreasing = TRUE)
  sorted_labels <- labels[ord]
  total_positives <- sum(labels == 1)
  if (total_positives == 0) return(NA)
  tp <- 0; fp <- 0
  precisions <- c(); recalls <- c()
  for (i in 1:n) {
    if (sorted_labels[i] == 1) { tp <- tp + 1 } else { fp <- fp + 1 }
    precisions <- c(precisions, tp / (tp + fp))
    recalls <- c(recalls, tp / total_positives)
  }
  auc <- 0
  for (i in 2:length(recalls)) {
    auc <- auc + (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2.0
  }
  return(auc)
}

# ============================================================================
# Max-MCC computation (sweep thresholds)
# ============================================================================
compute_max_mcc <- function(scores, labels) {
  thresholds <- seq(0.05, 0.95, by=0.01)
  best_mcc <- -2
  for (thr in thresholds) {
    pred <- ifelse(scores >= thr, 1, 0)
    tp <- sum(pred == 1 & labels == 1)
    tn <- sum(pred == 0 & labels == 0)
    fp <- sum(pred == 1 & labels == 0)
    fn <- sum(pred == 0 & labels == 1)
    denom <- sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if (denom == 0) next
    mcc <- (tp*tn - fp*fn) / denom
    if (mcc > best_mcc) best_mcc <- mcc
  }
  return(best_mcc)
}

# ============================================================================
# W1 Optimal Transport (Wasserstein-1 on RT deciles)
# ============================================================================
compute_w1 <- function(empirical_rt, predicted_rt) {
  n_deciles <- 10
  emp_q <- quantile(empirical_rt, probs = seq(0.1, 1.0, by=0.1), na.rm=TRUE)
  pred_q <- quantile(predicted_rt, probs = seq(0.1, 1.0, by=0.1), na.rm=TRUE)
  w1 <- mean(abs(emp_q - pred_q))
  return(w1)
}

# ============================================================================
# MAIN LOOP
# ============================================================================
cat("Running Phase 2/3/4 metrics for BVK and ECCM...\n")

all_results <- list()
for (p in participants) {
  print(paste("evaluating participant: ", p))
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0)
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) next

  # Fit BVK
  chain_23 <- run_mcmc_subject(23, iters, init_phi_23, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_23 <- eval_metrics_eccm_bvk(as.numeric(chain_23[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)

  # Fit ECCM baseline
  chain_6 <- run_mcmc_subject(6, iters, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_6 <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_6[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)

  # Build switch labels
  is_switch <- numeric(nrow(p_data))
  for (i in 2:nrow(p_data)) {
    is_switch[i] <- ifelse(p_data$Resp[i] != p_data$Resp[i-1], 1, 0)
  }
  idx <- 2:nrow(p_data)

  res_p <- list()
  for (model_name in c("bvk", "eccm")) {
    if (model_name == "bvk") {
      probs <- metrics_23$prob_ch1
      exp_rts <- metrics_23$exp_rt
    } else {
      probs <- metrics_6$prob_ch1
      exp_rts <- metrics_6$exp_rt
    }

    # P(switch) for each trial
    p_switch <- numeric(nrow(p_data))
    pred_stay <- numeric(nrow(p_data))
    for (i in idx) {
      pred_ch <- ifelse(probs[i] > 0.5, 1, 2)
      pred_stay[i] <- ifelse(pred_ch == p_data$Resp[i-1], 1, 0)
      if (p_data$Resp[i-1] == 1) {
        p_switch[i] <- 1.0 - probs[i]
      } else {
        p_switch[i] <- probs[i]
      }
    }

    # Confusion matrix
    actual_stay <- 1 - is_switch[idx]
    pred_is_stay <- pred_stay[idx]

    stay_acc <- sum(actual_stay == 1 & pred_is_stay == 1) / max(1, sum(actual_stay == 1))

    correct_switch <- 0
    for (i in idx) {
      if (is_switch[i] == 1 && pred_stay[i] == 0) correct_switch <- correct_switch + 1
    }

    pr_auc <- compute_pr_auc(p_switch[idx], is_switch[idx])
    max_mcc <- compute_max_mcc(p_switch[idx], is_switch[idx])

    # W1 optimal transport
    w1 <- compute_w1(p_data$RT, exp_rts)

    res_p[[model_name]] <- data.frame(
      subject = p,
      n_switches = sum(is_switch[idx] == 1),
      correct_switch = correct_switch,
      stay_acc = stay_acc,
      pr_auc = pr_auc,
      max_mcc = max_mcc,
      w1 = w1,
      stringsAsFactors = FALSE
    )
  }

  all_results[[as.character(p)]] <- res_p
  cat(sprintf("Done %s: BVK recall=%.1f%% MCC=%.3f W1=%.4f | ECCM recall=%.1f%% MCC=%.3f W1=%.4f\n",
    p,
    res_p$bvk$correct_switch / max(1, res_p$bvk$n_switches) * 100, res_p$bvk$max_mcc, res_p$bvk$w1,
    res_p$eccm$correct_switch / max(1, res_p$eccm$n_switches) * 100, res_p$eccm$max_mcc, res_p$eccm$w1))
}

# Aggregate
bvk_all <- do.call(rbind, lapply(all_results, function(x) x$bvk))
eccm_all <- do.call(rbind, lapply(all_results, function(x) x$eccm))

cat("\n=== FINAL PHASE 2/3 RESULTS ===\n")
cat(sprintf("BVK:  Switch Recall=%.2f%%  PR-AUC=%.4f  Max-MCC=%.4f  Stay Acc=%.2f%%\n",
  sum(bvk_all$correct_switch) / sum(bvk_all$n_switches) * 100,
  mean(bvk_all$pr_auc, na.rm=TRUE),
  mean(bvk_all$max_mcc, na.rm=TRUE),
  mean(bvk_all$stay_acc, na.rm=TRUE) * 100))
cat(sprintf("ECCM: Switch Recall=%.2f%%  PR-AUC=%.4f  Max-MCC=%.4f  Stay Acc=%.2f%%\n",
  sum(eccm_all$correct_switch) / sum(eccm_all$n_switches) * 100,
  mean(eccm_all$pr_auc, na.rm=TRUE),
  mean(eccm_all$max_mcc, na.rm=TRUE),
  mean(eccm_all$stay_acc, na.rm=TRUE) * 100))

cat("\n=== FINAL PHASE 4 RESULTS (W1 Optimal Transport) ===\n")
cat(sprintf("BVK:  Mean W1=%.4f (SD=%.4f)\n", mean(bvk_all$w1, na.rm=TRUE), sd(bvk_all$w1, na.rm=TRUE)))
cat(sprintf("ECCM: Mean W1=%.4f (SD=%.4f)\n", mean(eccm_all$w1, na.rm=TRUE), sd(eccm_all$w1, na.rm=TRUE)))
