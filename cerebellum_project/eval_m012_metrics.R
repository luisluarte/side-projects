library(cmdstanr)
library(posterior)
library(dplyr)
library(tidyr)
library(pROC)
library(PRROC)


cat("Loading Models and Data...\n")
df_n30 <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N30.rds")

# Load Fits
fit_vopt <- as_draws_df(read_cmdstan_csv(list.files("/home/DCCS5/cerebellum_project/results", pattern="fit_vopt_n30-.*\\.csv", full.names=TRUE))$post_warmup_draws)
fit_m012 <- as_draws_df(read_cmdstan_csv(list.files("/home/DCCS5/cerebellum_project/results", pattern="fit_m012_ctrl_n30-.*\\.csv", full.names=TRUE))$post_warmup_draws)

# Extract Median Parameters
get_median <- function(fit, param_name, N) {
  unlist(sapply(1:N, function(i) median(fit[[paste0(param_name, "[", i, "]")]])))
}

# V-OPT Params
v_base_vopt <- get_median(fit_vopt, "a_base_raw", 30)
v_tnd_vopt <- get_median(fit_vopt, "tnd", 30)
v_vctx_vopt <- get_median(fit_vopt, "v_ctx", 30)
v_aw_vopt <- get_median(fit_vopt, "aw", 30)
v_al_vopt <- get_median(fit_vopt, "al", 30)
v_wctx_vopt <- get_median(fit_vopt, "w_ctx", 30)
v_betamis_vopt <- get_median(fit_vopt, "beta_mismatch", 30)

# M012 Params
m_base_m012 <- get_median(fit_m012, "a_base_raw", 30)
m_tnd_m012 <- get_median(fit_m012, "tnd", 30)
m_vctx_m012 <- get_median(fit_m012, "v_ctx", 30)
m_aw_m012 <- get_median(fit_m012, "aw", 30)
m_al_m012 <- get_median(fit_m012, "al", 30)
m_apc_m012 <- get_median(fit_m012, "alpha_pc", 30)
m_tau_m012 <- get_median(fit_m012, "tau_decay", 30)
m_gs_m012 <- get_median(fit_m012, "golgi_scale", 30)
m_wcb_m012 <- get_median(fit_m012, "w_cb", 30)
m_wctx_m012 <- get_median(fit_m012, "w_ctx", 30)
m_betamis_m012 <- get_median(fit_m012, "beta_mismatch", 30)
m_frac <- 0.1 + 0.8 * (0:3 / 3.0); m_inv_frac <- 1.0 - m_frac

m_kappa <- 0.1 + 0.89 * (0:3 / 3.0)
W_exp <- matrix(rnorm(30 * 4, 0, 1), nrow=30, ncol=4) # Using the exact same W_exp as the showdown! 
set.seed(42) # Must set exact seed used in run_final_showdown_vopt.R before W_exp generation
W_exp <- matrix(rnorm(30 * 4, 0, 1), nrow=30, ncol=4)

cat("Simulating Trial-by-Trial Variables...\n")
log1p_exp <- function(x) { log1p(exp(x)) }

res_list <- list()

for (s in 1:30) {
  d_s <- df_n30 %>% filter(subj_idx == s)
  n <- nrow(d_s)
  
  # V-OPT State
  Q_v <- c(0.5, 0.5)
  phys_a_vopt <- 0.11 + 3.0 * (1/(1+exp(-v_base_vopt[s])))
  dmax_vopt <- 1.0 / phys_a_vopt
  
  # M012 State
  Q_m <- c(0.5, 0.5)
  phys_a_m012 <- 0.11 + 3.0 * (1/(1+exp(-m_base_m012[s])))
  dmax_m012 <- 1.0 / phys_a_m012
  frac_mem <- rep(0, 4)
  Z <- rep(0, 4)
  W_PC <- rep(0, 4)
  W_exp_s <- W_exp[s,]
  inv_W_exp <- m_inv_frac * W_exp_s
  inv_tau <- 1.0 / m_tau_m012[s]
  
  rt_pred_vopt <- numeric(n)
  ch_pred_vopt <- numeric(n)
  
  rt_pred_m012 <- numeric(n)
  ch_pred_m012 <- numeric(n)
  
  for (t in 1:n) {
    ch <- d_s$Boundary[t]
    R <- d_s$F[t]
    
    # ------------------
    # V-OPT Forward
    # ------------------
    Qdiff_v <- Q_v[1] - Q_v[2]
    M_align_v <- tanh(v_wctx_vopt[s] * Qdiff_v)
    caution_v <- log1p_exp(-5.0 * abs(M_align_v)) * 0.1
    a_v <- phys_a_vopt + dmax_vopt * tanh(v_betamis_vopt[s] * caution_v)
    veff_v <- v_vctx_vopt[s] * Qdiff_v
    
    rt_pred_vopt[t] <- v_tnd_vopt[s] + (a_v / (veff_v + 1e-8)) * tanh(a_v * veff_v)
    ch_pred_vopt[t] <- 1 / (1 + exp(-2 * a_v * veff_v))
    
    pe_v <- R - Q_v[ch]
    aeff_v <- ifelse(pe_v > 0, v_aw_vopt[s], v_al_vopt[s])
    Q_v[ch] <- Q_v[ch] + aeff_v * pe_v
    
    # ------------------
    # M012 Forward
    # ------------------
    phys_decay <- exp(-d_s$ITI[t] * inv_tau)
    Q_m <- 0.5 + (Q_m - 0.5) * phys_decay
    Qdiff_m <- Q_m[1] - Q_m[2]
    
    frac_mem <- m_frac * frac_mem + inv_W_exp * Q_m[ch]
    Z <- phys_decay * (m_kappa * Z) + tanh(frac_mem)
    
    W_PC_eff <- 3.0 * tanh(W_PC * (1/3))
    eff_z <- W_PC_eff * Z
    abs_app <- sqrt((eff_z * eff_z) + 1e-8)
    S_mask <- tanh(m_gs_m012[s] * abs_app)
    
    cb0 <- sum(S_mask[1:2] * eff_z[1:2])
    cb1 <- sum(S_mask[3:4] * eff_z[3:4])
    Cb_diff <- cb0 - cb1
    
    M_align_m <- tanh(m_wcb_m012[s] * Cb_diff) * tanh(m_wctx_m012[s] * Qdiff_m)
    caution_m <- log1p_exp(-10.0 * M_align_m) * 0.1
    a_m <- phys_a_m012 + dmax_m012 * tanh(m_betamis_m012[s] * caution_m)
    veff_m <- m_vctx_m012[s] * Qdiff_m
    
    rt_pred_m012[t] <- m_tnd_m012[s] + (a_m / (veff_m + 1e-8)) * tanh(a_m * veff_m)
    ch_pred_m012[t] <- 1 / (1 + exp(-2 * a_m * veff_m))
    
    pe_m <- R - Q_m[ch]
    aeff_m <- ifelse(pe_m > 0, m_aw_m012[s], m_al_m012[s])
    Q_m[ch] <- Q_m[ch] + aeff_m * pe_m
    
    alpha_E <- m_apc_m012[s] * pe_m
    if (ch == 1) { W_PC[1:2] <- W_PC[1:2] + alpha_E * Z[1:2] }
    else { W_PC[3:4] <- W_PC[3:4] + alpha_E * Z[3:4] }
  }
  
  d_s$rt_pred_vopt <- abs(rt_pred_vopt) # E[RT] can mathematically swing negative if a/v logic flips, abs handles drift symmetry
  d_s$ch_pred_vopt <- ch_pred_vopt
  d_s$rt_pred_m012 <- abs(rt_pred_m012)
  d_s$ch_pred_m012 <- ch_pred_m012
  
  res_list[[s]] <- d_s
}

df_res <- bind_rows(res_list)
# Remove NA and outliers
df_res <- df_res %>% filter(RT > 0, RT < 4.0)

# Calculate Metrics
calc_metrics <- function(truth, probs, rt_true, rt_pred) {
  truth_01 <- ifelse(truth == 1, 1, 0)
  roc_obj <- pROC::roc(truth_01, probs, quiet=TRUE)
  roc_auc <- as.numeric(pROC::auc(roc_obj))
  
  pr_obj <- PRROC::pr.curve(scores.class0 = probs[truth_01 == 1], scores.class1 = probs[truth_01 == 0], curve=FALSE)
  pr_auc <- pr_obj$auc.integral
  
  preds <- ifelse(probs >= 0.5, 1, 0)
  tp <- sum(preds == 1 & truth_01 == 1)
  tn <- sum(preds == 0 & truth_01 == 0)
  fp <- sum(preds == 1 & truth_01 == 0)
  fn <- sum(preds == 0 & truth_01 == 1)
  
  mcc_den <- sqrt(as.numeric(tp + fp) * as.numeric(tp + fn) * as.numeric(tn + fp) * as.numeric(tn + fn))
  mcc <- ifelse(mcc_den == 0, 0, ((tp * tn) - (fp * fn)) / mcc_den)
  
  rmse <- sqrt(mean((rt_true - rt_pred)^2))
  
  conf_mat <- matrix(c(tn, fp, fn, tp), nrow=2, byrow=TRUE)
  conf_norm <- conf_mat / sum(conf_mat)
  
  return(list(
    ROC_AUC = roc_auc,
    PR_AUC = pr_auc,
    MCC = mcc,
    RT_RMSE = rmse,
    CONF_NORM = conf_norm
  ))
}

met_vopt <- calc_metrics(df_res$Boundary, df_res$ch_pred_vopt, df_res$RT, df_res$rt_pred_vopt)
met_m012 <- calc_metrics(df_res$Boundary, df_res$ch_pred_m012, df_res$RT, df_res$rt_pred_m012)

# Write to text file
sink("/home/DCCS5/cerebellum_project/results/final_statistics.txt")
cat("=== STATISTICAL COMPARISON ===\n\n")

cat("V-OPT BASELINE:\n")
cat("ROC-AUC:", round(met_vopt$ROC_AUC, 4), "\n")
cat("PR-AUC :", round(met_vopt$PR_AUC, 4), "\n")
cat("MCC    :", round(met_vopt$MCC, 4), "\n")
cat("RT-RMSE:", round(met_vopt$RT_RMSE, 4), "\n")
cat("Normalized Confusion Matrix:\n")
print(round(met_vopt$CONF_NORM, 4))

cat("\nM012 CEREBELLAR RESERVOIR:\n")
cat("ROC-AUC:", round(met_m012$ROC_AUC, 4), "\n")
cat("PR-AUC :", round(met_m012$PR_AUC, 4), "\n")
cat("MCC    :", round(met_m012$MCC, 4), "\n")
cat("RT-RMSE:", round(met_m012$RT_RMSE, 4), "\n")
cat("Normalized Confusion Matrix:\n")
print(round(met_m012$CONF_NORM, 4))
sink()

cat("Saved to final_statistics.txt\n")
