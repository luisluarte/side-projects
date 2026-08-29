# ==============================================================================
# EXACT-R: STAGE 4 & 5 - EMPIRICAL DATA ANALYSIS & DISCRETE BASELINE FITTING
# Dataset: behavioral_compilate.csv (16,031 trials, 2AFC Probabilistic Reversal)
# Models: 1) Counterfactual EV Rescorla-Wagner, 2) Win-Stay Lose-Shift (WSLS)
# ==============================================================================
suppressPackageStartupMessages({
  library(stats)
  library(ggplot2)
})

cat("==============================================================================\n")
cat("STAGE 4 & 5: Empirical Ingestion & Probabilistic Baseline Model Fitting\n")
cat("==============================================================================\n\n")

dataset_path <- "../datasets/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

df_raw <- read.csv(dataset_path)
cat(sprintf("Loaded behavioral dataset: %d total trials across %d participants.\n",
            nrow(df_raw), length(unique(df_raw$participant_id))))

# --- DATA SUMMARY STATISTICS ---
participants <- unique(df_raw$participant_id)
N_sub <- length(participants)

cat(sprintf("Dataset Summary:\n"))
cat(sprintf("  Total Subjects:      %d\n", N_sub))
cat(sprintf("  Mean Trials/Subject: %.1f\n", nrow(df_raw) / N_sub))
cat(sprintf("  Mean Choice Acc:    %.2f%% (Optimal contingency choice)\n",
            mean(df_raw$Resp == (2 - df_raw$buena)) * 100))
cat(sprintf("  Overall Reward Rate:%.2f%%\n\n", mean(df_raw$F) * 100))

# --- MODEL 1: COUNTERFACTUAL EXPECTED VALUE RESCORLA-WAGNER (EV-RW) ---
# Parameters: par = c(logit_alpha, logit_alpha_c, log_beta)
fit_ev_rw_subject <- function(sub_df) {
  N <- nrow(sub_df)
  resp <- sub_df$Resp       # 1 or 2
  outcome <- sub_df$F       # 0 or 1
  m1 <- sub_df$Bd1          # Magnitude Option A
  m2 <- sub_df$Bd2          # Magnitude Option B
  
  nll_func <- function(par) {
    alpha <- 1 / (1 + exp(-par[1]))
    alpha_c <- 1 / (1 + exp(-par[2]))
    beta_sm <- exp(par[3])
    
    p_hat <- c(0.5, 0.5) # Initial beliefs
    nll <- 0.0
    
    for (t in 1:N) {
      # Expected values
      ev1 <- p_hat[1] * m1[t]
      ev2 <- p_hat[2] * m2[t]
      
      # Softmax choice probability for Option 1
      p_choice1 <- 1 / (1 + exp(-beta_sm * (ev1 - ev2)))
      p_choice1 <- pmax(1e-6, pmin(1 - 1e-6, p_choice1))
      
      ch <- resp[t]
      if (ch == 1) {
        nll <- nll - log(p_choice1)
        # Direct update chosen
        delta <- outcome[t] - p_hat[1]
        p_hat[1] <- p_hat[1] + alpha * delta
        # Counterfactual update unchosen
        p_hat[2] <- p_hat[2] + alpha_c * ((1 - outcome[t]) - p_hat[2])
      } else {
        nll <- nll - log(1 - p_choice1)
        # Direct update chosen
        delta <- outcome[t] - p_hat[2]
        p_hat[2] <- p_hat[2] + alpha * delta
        # Counterfactual update unchosen
        p_hat[1] <- p_hat[1] + alpha_c * ((1 - outcome[t]) - p_hat[1])
      }
    }
    return(nll)
  }
  
  opt <- optim(par = c(0, 0, 0), fn = nll_func, method = "BFGS")
  alpha_fit <- 1 / (1 + exp(-opt$par[1]))
  alpha_c_fit <- 1 / (1 + exp(-opt$par[2]))
  beta_fit <- exp(opt$par[3])
  nll <- opt$value
  
  k <- 3
  aic <- 2 * k + 2 * nll
  bic <- k * log(N) + 2 * nll
  
  return(c(alpha = alpha_fit, alpha_c = alpha_c_fit, beta = beta_fit, NLL = nll, AIC = aic, BIC = bic, N = N))
}

# --- MODEL 2: WIN-STAY LOSE-SHIFT (WSLS) ---
# Parameters: par = c(logit_p_win_stay, logit_p_lose_shift)
fit_wsls_subject <- function(sub_df) {
  N <- nrow(sub_df)
  resp <- sub_df$Resp
  outcome <- sub_df$F
  
  nll_func <- function(par) {
    p_ws <- 1 / (1 + exp(-par[1]))
    p_ls <- 1 / (1 + exp(-par[2]))
    
    nll <- 0.0
    for (t in 2:N) {
      prev_ch <- resp[t - 1]
      prev_out <- outcome[t - 1]
      curr_ch <- resp[t]
      
      if (prev_out == 1) {
        # Win scenario
        p_repeat <- p_ws
      } else {
        # Lose scenario
        p_repeat <- 1.0 - p_ls
      }
      
      p_choice <- if (curr_ch == prev_ch) p_repeat else (1.0 - p_repeat)
      p_choice <- pmax(1e-6, pmin(1 - 1e-6, p_choice))
      nll <- nll - log(p_choice)
    }
    return(nll)
  }
  
  opt <- optim(par = c(0, 0), fn = nll_func, method = "BFGS")
  p_ws_fit <- 1 / (1 + exp(-opt$par[1]))
  p_ls_fit <- 1 / (1 + exp(-opt$par[2]))
  nll <- opt$value
  
  k <- 2
  aic <- 2 * k + 2 * nll
  bic <- k * log(N - 1) + 2 * nll
  
  return(c(p_win_stay = p_ws_fit, p_lose_shift = p_ls_fit, NLL = nll, AIC = aic, BIC = bic, N = N - 1))
}

cat("Fitting Discrete Baseline Models across all participants... ")
res_ev_rw <- t(sapply(participants, function(s) fit_ev_rw_subject(df_raw[df_raw$participant_id == s, ])))
res_wsls  <- t(sapply(participants, function(s) fit_wsls_subject(df_raw[df_raw$participant_id == s, ])))
cat("Done!\n\n")

# Aggregate Fit Results
mean_ev_rw <- colMeans(res_ev_rw)
mean_wsls <- colMeans(res_wsls)

cat("==============================================================================\n")
cat("MODEL FITTING BENCHMARK SUMMARY:\n")
cat("==============================================================================\n")
cat(sprintf("1) COUNTERFACTUAL EV RESCORLA-WAGNER MODEL:\n"))
cat(sprintf("   Learning Rate (alpha):           %.4f\n", mean_ev_rw["alpha"]))
cat(sprintf("   Counterfactual LR (alpha_c):     %.4f\n", mean_ev_rw["alpha_c"]))
cat(sprintf("   Softmax Inverse Temp (beta):      %.4f\n", mean_ev_rw["beta"]))
cat(sprintf("   Mean NLL per Subject:            %.2f\n", mean_ev_rw["NLL"]))
cat(sprintf("   Total AIC / BIC:                 %.1f / %.1f\n\n", sum(res_ev_rw[, "AIC"]), sum(res_ev_rw[, "BIC"])))

cat(sprintf("2) WIN-STAY LOSE-SHIFT (WSLS) MODEL:\n"))
cat(sprintf("   Win-Stay Probability (p_ws):     %.4f\n", mean_wsls["p_win_stay"]))
cat(sprintf("   Lose-Shift Probability (p_ls):   %.4f\n", mean_wsls["p_lose_shift"]))
cat(sprintf("   Mean NLL per Subject:            %.2f\n", mean_wsls["NLL"]))
cat(sprintf("   Total AIC / BIC:                 %.1f / %.1f\n", sum(res_wsls[, "AIC"]), sum(res_wsls[, "BIC"])))
cat("==============================================================================\n\n")

# Save summary results CSV
baseline_summary <- data.frame(
  Model = c("Counterfactual_EV_RW", "Win_Stay_Lose_Shift"),
  Total_NLL = c(sum(res_ev_rw[, "NLL"]), sum(res_wsls[, "NLL"])),
  Total_AIC = c(sum(res_ev_rw[, "AIC"]), sum(res_wsls[, "AIC"])),
  Total_BIC = c(sum(res_ev_rw[, "BIC"]), sum(res_wsls[, "BIC"])),
  Mean_NLL_per_Sub = c(mean_ev_rw["NLL"], mean_wsls["NLL"])
)
write.csv(baseline_summary, "baseline_models_benchmark.csv", row.names = FALSE)
cat("Saved baseline model summary to baseline_models_benchmark.csv\n")
