library(Rcpp)
library(doParallel)
library(foreach)

# ============================================================================
# Q-Learning with Counterfactual Update — implemented in pure R for simplicity
# 4 params: alpha (learning rate), beta (inverse temp), alpha_c (counterfactual LR), bias
# ============================================================================
q_learning_counterfactual <- function(resp, out, alpha, beta, alpha_c, bias) {
  Q <- c(0.0, 0.0)  # Q[1], Q[2]
  prob_ch1 <- numeric(length(resp))
  
  for (t in 1:length(resp)) {
    # Softmax choice probability
    prob_ch1[t] <- 1.0 / (1.0 + exp(-beta * (Q[1] - Q[2]) - bias))
    
    ch <- resp[t]
    R <- ifelse(out[t] == 1, 1.0, 0.0)
    
    # Chosen update
    Q[ch] <- Q[ch] + alpha * (R - Q[ch])
    
    # Counterfactual update: unchosen gets updated with (1 - R)
    unch <- ifelse(ch == 1, 2, 1)
    Q[unch] <- Q[unch] + alpha_c * ((1.0 - R) - Q[unch])
  }
  
  return(prob_ch1)
}

# Grid search for Q-learning (no MCMC needed for this simple model)
fit_qlearning <- function(resp, out) {
  best_nll <- Inf
  best_params <- c(0.1, 1.0, 0.1, 0.0)
  
  for (alpha in c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5)) {
    for (beta_val in c(0.5, 1.0, 2.0, 3.0, 5.0)) {
      for (alpha_c in c(0.0, 0.01, 0.05, 0.1, 0.2)) {
        for (bias in c(-0.5, 0.0, 0.5)) {
          probs <- q_learning_counterfactual(resp, out, alpha, beta_val, alpha_c, bias)
          # Negative log-likelihood
          nll <- 0
          for (t in 1:length(resp)) {
            p <- ifelse(resp[t] == 1, probs[t], 1.0 - probs[t])
            p <- max(p, 1e-8)
            nll <- nll - log(p)
          }
          if (nll < best_nll) {
            best_nll <- nll
            best_params <- c(alpha, beta_val, alpha_c, bias)
          }
        }
      }
    }
  }
  
  probs <- q_learning_counterfactual(resp, out, best_params[1], best_params[2], best_params[3], best_params[4])
  return(list(prob_ch1 = probs, nll = best_nll, params = best_params))
}

# ============================================================================
# WSLS metrics (pure R, mirrors the C++ logic)
# ============================================================================
wsls_metrics <- function(resp, out) {
  prob_ch1 <- numeric(length(resp))
  prob_ch1[1] <- 0.5
  for (t in 2:length(resp)) {
    if (out[t-1] == 1) {
      # Win-Stay: predict same as last choice
      prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.999, 0.001)
    } else {
      # Lose-Switch: predict opposite of last choice
      prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.001, 0.999)
    }
  }
  return(prob_ch1)
}

# ============================================================================
# PR-AUC computation (manual, no external package dependency)
# ============================================================================
compute_pr_auc <- function(scores, labels) {
  # labels: 1 = switch (positive), 0 = stay (negative)
  # scores: P(switch) for each trial
  n <- length(scores)
  
  # Sort by decreasing score
  ord <- order(scores, decreasing = TRUE)
  sorted_labels <- labels[ord]
  
  total_positives <- sum(labels == 1)
  if (total_positives == 0) return(NA)
  
  tp <- 0; fp <- 0
  precisions <- c()
  recalls <- c()
  
  for (i in 1:n) {
    if (sorted_labels[i] == 1) {
      tp <- tp + 1
    } else {
      fp <- fp + 1
    }
    precision <- tp / (tp + fp)
    recall <- tp / total_positives
    precisions <- c(precisions, precision)
    recalls <- c(recalls, recall)
  }
  
  # Trapezoidal AUC over recall
  auc <- 0
  for (i in 2:length(recalls)) {
    auc <- auc + (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2.0
  }
  return(auc)
}

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 75

# Model 6 (Baseline ECCM Cortical RPE): 8 params
init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))
# Model 18 (MF-Gated Choice Reversal): 11 params
init_phi_18 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(1.0))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running 4-model comparison: WSLS vs Q-Learning(CF) vs ECCM vs MF-Reversal...\n")
results_list <- foreach(p = participants, .packages = c("Rcpp"), .export = c("q_learning_counterfactual", "fit_qlearning", "wsls_metrics", "compute_pr_auc")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi_reversal.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # ---- Model 1: WSLS ----
  probs_wsls <- wsls_metrics(p_data$Resp, p_data$F)
  
  # ---- Model 2: Q-Learning with Counterfactual Update ----
  ql_fit <- fit_qlearning(p_data$Resp, p_data$F)
  probs_ql <- ql_fit$prob_ch1
  
  # ---- Model 3: Baseline ECCM (Cortical RPE, Model 6) ----
  chain_6 <- run_mcmc_subject(6, iters, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_6 <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_6[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  probs_eccm <- metrics_6$prob_ch1
  
  # ---- Model 4: MF-Gated Choice Reversal (Model 18) ----
  chain_18 <- run_mcmc_subject(18, iters, init_phi_18, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_18 <- eval_metrics_eccm_golgi_reversal(as.numeric(chain_18[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  probs_rev <- metrics_18$prob_ch1
  
  # ---- Compute metrics for each model ----
  results <- list()
  model_names <- c("wsls", "ql", "eccm", "reversal")
  all_probs <- list(probs_wsls, probs_ql, probs_eccm, probs_rev)
  
  # Build labels: is_switch[i] = 1 if trial i is a switch from trial i-1
  is_switch <- numeric(nrow(p_data))
  for (i in 2:nrow(p_data)) {
    is_switch[i] <- ifelse(p_data$Resp[i] != p_data$Resp[i-1], 1, 0)
  }
  
  for (m in 1:4) {
    probs <- all_probs[[m]]
    
    # Predicted choice and stay/switch
    pred_stay <- numeric(nrow(p_data))
    # P(switch) = probability of choosing the opposite of previous choice
    p_switch <- numeric(nrow(p_data))
    
    for (i in 2:nrow(p_data)) {
      pred_ch <- ifelse(probs[i] > 0.5, 1, 2)
      pred_stay[i] <- ifelse(pred_ch == p_data$Resp[i-1], 1, 0)
      
      # P(switch) = P(choosing the opposite of previous)
      if (p_data$Resp[i-1] == 1) {
        p_switch[i] <- 1.0 - probs[i]  # P(choosing 2)
      } else {
        p_switch[i] <- probs[i]  # P(choosing 1)
      }
    }
    
    # Confusion matrix (trials 2:end only)
    idx <- 2:nrow(p_data)
    actual_stay <- 1 - is_switch[idx]
    pred_is_stay <- pred_stay[idx]
    
    TP <- sum(actual_stay == 1 & pred_is_stay == 1) / max(1, sum(actual_stay == 1))
    FN <- sum(actual_stay == 1 & pred_is_stay == 0) / max(1, sum(actual_stay == 1))
    FP <- sum(is_switch[idx] == 1 & pred_is_stay == 1) / max(1, sum(is_switch[idx] == 1))
    TN <- sum(is_switch[idx] == 1 & pred_is_stay == 0) / max(1, sum(is_switch[idx] == 1))
    
    # PR-AUC with switch as positive class
    pr_auc <- compute_pr_auc(p_switch[idx], is_switch[idx])
    
    # Raw counts for aggregate recall
    n_switches <- sum(is_switch[idx] == 1)
    n_pred_switch <- sum(is_switch[idx] == 1 & pred_is_stay[which(is_switch[idx] == 1)] == 0, na.rm=TRUE)
    # More robust: count directly
    correct_switch <- 0
    for (i in idx) {
      if (is_switch[i] == 1 && pred_stay[i] == 0) correct_switch <- correct_switch + 1
    }
    
    results[[model_names[m]]] <- data.frame(
      subject = p,
      TP = TP, FN = FN, FP = FP, TN = TN,
      pr_auc = pr_auc,
      n_switches = n_switches,
      correct_switch = correct_switch,
      stringsAsFactors = FALSE
    )
  }
  
  results
}

stopCluster(cl)

# Aggregate results
models <- c("wsls", "ql", "eccm", "reversal")
model_labels <- c("WSLS", "Q-Learning (CF)", "Baseline ECCM", "MF-Gated Reversal")
agg <- list()

for (m in models) {
  rows <- lapply(results_list, function(x) if (!is.null(x)) x[[m]] else NULL)
  rows <- do.call(rbind, rows)
  agg[[m]] <- rows
}

format_cell <- function(vec) {
  sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)
}

# Build report
report <- c(
  "# 4-Model Comparison: Confusion Matrices & PR-AUC",
  "",
  "Comparing **WSLS**, **Q-Learning with Counterfactual Update**, **Baseline ECCM (Cortical RPE)**, and **MF-Gated Choice Reversal** (Model 18).",
  "",
  "Switch is the **positive class** for PR-AUC.",
  ""
)

for (i in 1:4) {
  m <- models[i]
  d <- agg[[m]]
  recall <- sum(d$correct_switch) / sum(d$n_switches) * 100
  
  report <- c(report,
    sprintf("## %s", model_labels[i]),
    "",
    "| Actual \\ Predicted | Predict Stay | Predict Switch |",
    "| :--- | :--- | :--- |",
    sprintf("| **Actual Stay** | %s | %s |", format_cell(d$TP), format_cell(d$FN)),
    sprintf("| **Actual Switch** | %s | %s |", format_cell(d$FP), format_cell(d$TN)),
    "",
    sprintf("*   **Aggregate Switch Recall:** %.2f%%", recall),
    sprintf("*   **Mean PR-AUC (Switch+):** %.4f (±%.4f)", mean(d$pr_auc, na.rm=TRUE), sd(d$pr_auc, na.rm=TRUE)),
    ""
  )
}

# Summary table
report <- c(report,
  "---",
  "",
  "## Summary Table",
  "",
  "| Model | Switch Recall | PR-AUC (Switch+) | Stay Accuracy |",
  "| :--- | :---: | :---: | :---: |"
)

for (i in 1:4) {
  m <- models[i]
  d <- agg[[m]]
  recall <- sum(d$correct_switch) / sum(d$n_switches) * 100
  pr_auc_mean <- mean(d$pr_auc, na.rm=TRUE)
  stay_acc <- mean(d$TP, na.rm=TRUE) * 100
  
  report <- c(report, sprintf("| **%s** | %.2f%% | %.4f | %.2f%% |", model_labels[i], recall, pr_auc_mean, stay_acc))
}

writeLines(report, "docs/Four_Model_Comparison.md")
cat("\nReport successfully generated in docs/Four_Model_Comparison.md\n")
