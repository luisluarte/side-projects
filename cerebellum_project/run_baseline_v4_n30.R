library(cmdstanr)
library(posterior)
library(dplyr)
library(loo)
library(pROC)
library(PRROC)
library(lme4)

set.seed(42)
options(mc.cores = 8)

cat("==========================================\n")
cat(" RUNNING BASELINE V2 (N=30) AND FINAL METRICS\n")
cat("==========================================\n\n")

df_n30 <- readRDS("/home/DCCS5/cerebellum_project/data/processed/urgency_dat_N30.rds")
N_trials <- nrow(df_n30)
N_subj <- 30
W_exp <- matrix(rnorm(N_subj * 4, 0, 1), nrow=N_subj, ncol=4)
start_idx <- integer(N_subj)
end_idx <- integer(N_subj)
for(s in 1:N_subj) {
  start_idx[s] <- min(which(df_n30$subj_idx == s))
  end_idx[s] <- max(which(df_n30$subj_idx == s))
}

stan_data_v4 <- list(
  N_trials = N_trials,
  N_subj = N_subj,
  subj = df_n30$subj_idx,
  resp = df_n30$Boundary,
  reward = df_n30$F,
  rt = df_n30$RT,
  iti = df_n30$ITI,
  min_rt = df_n30 %>% group_by(subj_idx) %>% summarise(m = min(RT)) %>% pull(m),
  W_exp = W_exp,
  start_idx = start_idx,
  end_idx = end_idx,
  theta_mean = rep(0, 8),
  L_Sigma = diag(8),
  grainsize = 1
)

cat("Compiling Baseline V2...\n")
mod_v4 <- cmdstan_model("/home/DCCS5/cerebellum_project/src/stan/baseline_v4.stan", cpp_options = list(stan_threads = TRUE))

cat("Sampling Baseline V2 (N=30) ...\n")
fit_v4 <- mod_v4$sample(
  data = stan_data_v4,
  chains = 4,
  parallel_chains = 2,
  threads_per_chain = 4,
  init = 0,
  iter_warmup = 1000,
  iter_sampling = 1000,
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 10
)
fit_v4$save_output_files(dir = "/home/DCCS5/cerebellum_project/results/", basename = "fit_v4_n30")

summ_v2 <- fit_v4$summary()
cat("Baseline V2 Diagnostics -> Max Rhat:", max(summ_v2$rhat, na.rm=TRUE), "| Min Bulk ESS:", min(summ_v2$ess_bulk, na.rm=TRUE), "\n")
loo_v4 <- fit_v4$loo()
saveRDS(loo_v4, "/home/DCCS5/cerebellum_project/results/loo_v4_n30.rds")

loo_m012 <- readRDS("/home/DCCS5/cerebellum_project/results/loo_m012_n30.rds")
cat("\n==========================================\n")
cat(" LOO-CV COMPARISON (N=30)\n")
cat("==========================================\n")
comp <- loo_compare(list(M012 = loo_m012, BaselineV4 = loo_v4))
print(comp)
cat("\n")

cat("\nCalculating Generative Predictive Metrics for M012...\n")
# Load the perfectly saved M012 fit
csv_files <- list.files("/home/DCCS5/cerebellum_project/results", pattern="fit_m012_n30.*\\.csv", full.names=TRUE)
fit_m012 <- as_cmdstan_fit(csv_files)
dr <- as_draws_df(fit_m012$draws())
get_med <- function(name) median(dr[[name]])
get_vec_med <- function(name, len) sapply(1:len, function(i) median(dr[[paste0(name, "[", i, "]")]]))
get_mat_med <- function(name, rows, cols) {
  mat <- matrix(0, nrow=rows, ncol=cols)
  for(r in 1:rows) for(c in 1:cols) mat[r,c] <- median(dr[[paste0(name, "[", r, ",", c, "]")]])
  mat
}

mu_a_unc <- get_med("theta_unc[1]")
mu_tnd_unc <- get_med("theta_unc[2]")
mu_v_unc <- get_med("theta_unc[3]")
mu_res_raw <- get_vec_med("theta_unc", 11)[4:11]
sigma <- get_vec_med("sigma", 11)
z <- get_mat_med("z", 11, N_subj)
inv_logit <- function(x) 1/(1+exp(-x))
log1p_exp <- function(x) log1p(exp(x))
a_base_raw <- mu_a_unc + sigma[1]*z[1,]
tnd_cap <- pmin(stan_data_v4$min_rt - 0.05, 3.69)
tnd <- 0.01 + (tnd_cap - 0.01)*inv_logit(mu_tnd_unc + sigma[2]*z[2,])
v_ctx <- 18.51*inv_logit(mu_v_unc + sigma[3]*z[3,])
aw <- inv_logit(mu_res_raw[1] + sigma[4]*z[4,])
al <- inv_logit(mu_res_raw[2] + sigma[5]*z[5,])
alpha_pc <- inv_logit(mu_res_raw[3] + sigma[6]*z[6,])
tau_decay <- log1p_exp(mu_res_raw[4] + sigma[7]*z[7,])
golgi_scale <- log1p_exp(mu_res_raw[5] + sigma[8]*z[8,])
w_cb <- log1p_exp(mu_res_raw[6] + sigma[9]*z[9,])
w_ctx <- log1p_exp(mu_res_raw[7] + sigma[10]*z[10,])
beta_mismatch <- log1p_exp(mu_res_raw[8] + sigma[11]*z[11,])
frac_alpha <- 0.1 + 0.8*(0:3)/3.0
inv_frac_alpha <- 1.0 - frac_alpha
kappa_vec <- 0.1 + 0.89*(0:3)/3.0
clean_iti <- ifelse(df_n30$ITI < 0, 1.0, df_n30$ITI)
pred_ch <- numeric(N_trials)
pred_prob <- numeric(N_trials)
pred_rt <- numeric(N_trials)
for(s in 1:N_subj) {
  Q <- c(0.5, 0.5)
  frac_mem <- rep(0, 4)
  Z <- rep(0, 4)
  W_PC_latent <- rep(0, 4)
  inv_W_exp <- inv_frac_alpha * W_exp[s,]
  phys_a_base <- 0.11 + 3.0*inv_logit(a_base_raw[s])
  delta_max <- 1.0/phys_a_base
  inv_tau <- 1.0/tau_decay[s]
  for(t in start_idx[s]:end_idx[s]) {
    ch <- df_n30$Boundary[t]
    R <- df_n30$F[t]
    phys_decay <- exp(-clean_iti[t]*inv_tau)
    Q <- 0.5 + (Q - 0.5)*phys_decay
    Q_diff <- Q[1] - Q[2]
    frac_mem <- frac_alpha*frac_mem + inv_W_exp*Q[ch]
    Z <- phys_decay*(kappa_vec*Z) + tanh(frac_mem)
    W_PC_eff <- 3.0*tanh(W_PC_latent/3.0)
    eff_z <- W_PC_eff*Z
    abs_approx <- sqrt(eff_z^2 + 1e-8)
    S_mask <- tanh(golgi_scale[s]*abs_approx)
    cb0 <- sum(S_mask[1:2]*eff_z[1:2])
    cb1 <- sum(S_mask[3:4]*eff_z[3:4])
    Cb_diff <- cb0 - cb1
    M_align <- tanh(w_cb[s]*Cb_diff)*tanh(w_ctx[s]*Q_diff)
    caution <- log1p_exp(-10.0*M_align)*0.1
    a_dyn <- phys_a_base + delta_max*tanh(beta_mismatch[s]*caution)
    veff_raw <- v_ctx[s]*Q_diff
    pred_prob[t] <- inv_logit(veff_raw)
    pred_ch[t] <- ifelse(veff_raw > 0, 1, 2)
    if(abs(veff_raw) > 1e-6) {
        pred_rt[t] <- tnd[s] + (a_dyn / (2*abs(veff_raw))) * tanh(abs(veff_raw)*a_dyn)
    } else {
        pred_rt[t] <- tnd[s] + (a_dyn^2)/2
    }
    pe <- R - Q[ch]
    alpha_eff <- ifelse(pe>0, aw[s], al[s])
    Q[ch] <- Q[ch] + alpha_eff*pe
    alpha_E <- alpha_pc[s]*pe
    if(ch == 1) { W_PC_latent[1:2] <- W_PC_latent[1:2] + alpha_E*Z[1:2] }
    else { W_PC_latent[3:4] <- W_PC_latent[3:4] + alpha_E*Z[3:4] }
  }
}

y_true <- ifelse(df_n30$Boundary == 1, 1, 0)
roc <- roc(y_true, pred_prob, quiet=TRUE)
pr <- pr.curve(scores.class0 = pred_prob, weights.class0 = y_true)
valid_rt <- !is.na(df_n30$RT)
rmse <- sqrt(mean((df_n30$RT[valid_rt] - pred_rt[valid_rt])^2))
mcc_num <- sum(y_true == 1 & pred_ch == 1)*sum(y_true == 0 & pred_ch == 2) - sum(y_true == 0 & pred_ch == 1)*sum(y_true == 1 & pred_ch == 2)
mcc_den <- sqrt(sum(y_true == 1)*sum(y_true == 0)*sum(pred_ch == 1)*sum(pred_ch == 2))
mcc <- mcc_num / mcc_den

cat("\n==========================================\n")
cat(" M012 FINAL METRICS (N=30)\n")
cat("==========================================\n")
cat("ROC-AUC:", roc$auc, "\n")
cat("PR-AUC :", pr$auc.integral, "\n")
cat("RT-RMSE:", rmse, "\n")
cat("MCC    :", mcc, "\n")

df_lmer <- data.frame(subj = factor(df_n30$subj_idx), y_true = y_true, y_pred = pred_prob)
m <- glmer(y_true ~ y_pred + (1|subj), data=df_lmer, family=binomial)
cat("\nLMER M012:\n")
print(summary(m)$coefficients)
cat("\nShowdown complete!\n")
