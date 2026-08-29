library(Rcpp)
library(doParallel)
library(foreach)

# PR-AUC computation
compute_pr_auc <- function(scores, labels) {
  n <- length(scores)
  ord <- order(scores, decreasing = TRUE)
  sorted_labels <- labels[ord]
  total_positives <- sum(labels == 1)
  if (total_positives == 0) return(NA)
  tp <- 0; fp <- 0; precisions <- c(); recalls <- c()
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

# WSLS metrics
wsls_metrics <- function(resp, out) {
  prob_ch1 <- numeric(length(resp)); prob_ch1[1] <- 0.5
  for (t in 2:length(resp)) {
    if (out[t-1] == 1) { prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.999, 0.001) }
    else { prob_ch1[t] <- ifelse(resp[t-1] == 1, 0.001, 0.999) }
  }
  return(prob_ch1)
}

# Q-Learning with Counterfactual Update
q_learning_counterfactual <- function(resp, out, alpha, beta, alpha_c, bias) {
  Q <- c(0.0, 0.0); prob_ch1 <- numeric(length(resp))
  for (t in 1:length(resp)) {
    prob_ch1[t] <- 1.0 / (1.0 + exp(-beta * (Q[1] - Q[2]) - bias))
    ch <- resp[t]; R <- ifelse(out[t] == 1, 1.0, 0.0)
    Q[ch] <- Q[ch] + alpha * (R - Q[ch])
    unch <- ifelse(ch == 1, 2, 1)
    Q[unch] <- Q[unch] + alpha_c * ((1.0 - R) - Q[unch])
  }
  return(prob_ch1)
}

fit_qlearning <- function(resp, out) {
  best_nll <- Inf; best_params <- c(0.1, 1.0, 0.1, 0.0)
  for (alpha in c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5)) {
    for (beta_val in c(0.5, 1.0, 2.0, 3.0, 5.0)) {
      for (alpha_c in c(0.0, 0.01, 0.05, 0.1, 0.2)) {
        for (bias in c(-0.5, 0.0, 0.5)) {
          probs <- q_learning_counterfactual(resp, out, alpha, beta_val, alpha_c, bias)
          nll <- 0
          for (t in 1:length(resp)) {
            p <- ifelse(resp[t] == 1, probs[t], 1.0 - probs[t])
            nll <- nll - log(max(p, 1e-8))
          }
          if (nll < best_nll) { best_nll <- nll; best_params <- c(alpha, beta_val, alpha_c, bias) }
        }
      }
    }
  }
  probs <- q_learning_counterfactual(resp, out, best_params[1], best_params[2], best_params[3], best_params[4])
  return(list(prob_ch1 = probs, params = best_params))
}

# Helper to compute per-subject metrics from prob_ch1
compute_subject_metrics <- function(probs, resp) {
  n <- length(resp)
  is_switch <- numeric(n); pred_stay <- numeric(n); p_switch <- numeric(n)
  for (i in 2:n) {
    is_switch[i] <- ifelse(resp[i] != resp[i-1], 1, 0)
    pred_ch <- ifelse(probs[i] > 0.5, 1, 2)
    pred_stay[i] <- ifelse(pred_ch == resp[i-1], 1, 0)
    if (resp[i-1] == 1) { p_switch[i] <- 1.0 - probs[i] } else { p_switch[i] <- probs[i] }
  }
  idx <- 2:n
  actual_stay <- 1 - is_switch[idx]
  TP <- sum(actual_stay == 1 & pred_stay[idx] == 1) / max(1, sum(actual_stay == 1))
  FN <- sum(actual_stay == 1 & pred_stay[idx] == 0) / max(1, sum(actual_stay == 1))
  FP <- sum(is_switch[idx] == 1 & pred_stay[idx] == 1) / max(1, sum(is_switch[idx] == 1))
  TN <- sum(is_switch[idx] == 1 & pred_stay[idx] == 0) / max(1, sum(is_switch[idx] == 1))
  pr_auc <- compute_pr_auc(p_switch[idx], is_switch[idx])
  correct_switch <- sum(is_switch[idx] == 1 & pred_stay[idx] == 0)
  n_switches <- sum(is_switch[idx] == 1)
  data.frame(TP=TP, FN=FN, FP=FP, TN=TN, pr_auc=pr_auc, correct_switch=correct_switch, n_switches=n_switches)
}

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 75
init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))
init_phi_18 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(1.0))
# Model 19: 12 params (splits explore_gain into win/loss)
init_phi_19 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), log(0.1), 0.0, log(0.1), log(2.0))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running 5-model comparison: WSLS vs QL(CF) vs ECCM vs Reversal vs AsymReversal...\n")
results_list <- foreach(p = participants, .packages = c("Rcpp"), .export = c("q_learning_counterfactual", "fit_qlearning", "wsls_metrics", "compute_pr_auc", "compute_subject_metrics")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi_reversal.cpp")
  sourceCpp("src/models/evaluate_metrics_golgi_asym_reversal.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) return(NULL)
  
  # 1. WSLS
  m_wsls <- compute_subject_metrics(wsls_metrics(p_data$Resp, p_data$F), p_data$Resp)
  
  # 2. Q-Learning CF
  ql_fit <- fit_qlearning(p_data$Resp, p_data$F)
  m_ql <- compute_subject_metrics(ql_fit$prob_ch1, p_data$Resp)
  
  # 3. Baseline ECCM
  chain_6 <- run_mcmc_subject(6, iters, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_6 <- eval_metrics_eccm_cortical_rpe(as.numeric(chain_6[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  m_eccm <- compute_subject_metrics(metrics_6$prob_ch1, p_data$Resp)
  
  # 4. Model 18 (Symmetric Reversal)
  chain_18 <- run_mcmc_subject(18, iters, init_phi_18, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_18 <- eval_metrics_eccm_golgi_reversal(as.numeric(chain_18[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_18 <- eval_eccm_golgi_reversal(as.numeric(chain_18[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  m_rev <- compute_subject_metrics(metrics_18$prob_ch1, p_data$Resp)
  
  # 5. Model 19 (Asymmetric Reversal)
  chain_19 <- run_mcmc_subject(19, iters, init_phi_19, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  metrics_19 <- eval_metrics_eccm_golgi_asym_reversal(as.numeric(chain_19[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_19 <- eval_eccm_golgi_asym_reversal(as.numeric(chain_19[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  m_asym <- compute_subject_metrics(metrics_19$prob_ch1, p_data$Resp)
  
  list(
    wsls = cbind(subject=p, m_wsls),
    ql = cbind(subject=p, m_ql),
    eccm = cbind(subject=p, m_eccm),
    rev = cbind(subject=p, m_rev, dev=dev_18),
    asym = cbind(subject=p, m_asym, dev=dev_19,
                 theta_win=exp(chain_19[iters, 11]),
                 theta_loss=exp(chain_19[iters, 12]))
  )
}

stopCluster(cl)

models <- c("wsls", "ql", "eccm", "rev", "asym")
model_labels <- c("WSLS", "Q-Learning (CF)", "Baseline ECCM", "MF-Reversal (Sym)", "MF-Reversal (Asym)")
agg <- list()
for (m in models) {
  rows <- lapply(results_list, function(x) if (!is.null(x)) x[[m]] else NULL)
  agg[[m]] <- do.call(rbind, rows)
}

format_cell <- function(vec) sprintf("%.2f%% (±%.2f%%)", mean(vec, na.rm=TRUE)*100, sd(vec, na.rm=TRUE)*100)

report <- c(
  "# 5-Model Comparison: Asymmetric Exploration",
  "",
  "Does splitting θ_explore into separate **win** and **loss** gains improve switch prediction and PR-AUC?",
  ""
)

for (i in 1:5) {
  m <- models[i]; d <- agg[[m]]
  recall <- sum(d$correct_switch) / sum(d$n_switches) * 100
  report <- c(report,
    sprintf("## %s", model_labels[i]), "",
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

# Asymmetric params
d_asym <- agg[["asym"]]
d_rev <- agg[["rev"]]

report <- c(report,
  "---", "",
  "## Summary Table", "",
  "| Model | Switch Recall | PR-AUC (Switch+) | Stay Accuracy |",
  "| :--- | :---: | :---: | :---: |"
)

for (i in 1:5) {
  m <- models[i]; d <- agg[[m]]
  recall <- sum(d$correct_switch) / sum(d$n_switches) * 100
  report <- c(report, sprintf("| **%s** | %.2f%% | %.4f | %.2f%% |",
    model_labels[i], recall, mean(d$pr_auc, na.rm=TRUE), mean(d$TP, na.rm=TRUE)*100))
}

report <- c(report, "",
  "## Asymmetric Parameters", "",
  sprintf("*   **θ_explore_win (median):** %.4f", median(d_asym$theta_win, na.rm=TRUE)),
  sprintf("*   **θ_explore_loss (median):** %.4f", median(d_asym$theta_loss, na.rm=TRUE)),
  sprintf("*   **Loss/Win Ratio:** %.2fx", median(d_asym$theta_loss, na.rm=TRUE) / max(1e-6, median(d_asym$theta_win, na.rm=TRUE))),
  "",
  "## Deviance Comparison (Model 18 vs 19)", "",
  sprintf("*   **Model 18 (Symmetric) Mean Deviance:** %.2f", mean(d_rev$dev, na.rm=TRUE)),
  sprintf("*   **Model 19 (Asymmetric) Mean Deviance:** %.2f", mean(d_asym$dev, na.rm=TRUE)),
  sprintf("*   **Paired t-test p-value:** %.4e", t.test(d_rev$dev, d_asym$dev, paired=TRUE)$p.value)
)

writeLines(report, "docs/Asymmetric_Reversal_Results.md")
cat("\nReport successfully generated in docs/Asymmetric_Reversal_Results.md\n")
