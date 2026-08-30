library(cmdstanr)
library(dplyr)
library(readr)
library(loo)
library(lme4)
library(lmerTest)

cat("1. Loading CSVs...\n")
base_csvs <- list.files(path = "/tmp", pattern = "baseline-.*\\.csv$", full.names = TRUE, recursive = TRUE)
m006_csvs <- list.files(path = "/tmp", pattern = "m006_strict_hmc-.*\\.csv$", full.names = TRUE, recursive = TRUE)

# Sort by modification time to get the latest 4 chains
base_csvs <- head(base_csvs[order(file.info(base_csvs)$mtime, decreasing = TRUE)], 4)
m006_csvs <- head(m006_csvs[order(file.info(m006_csvs)$mtime, decreasing = TRUE)], 4)

cat("Reading Baseline...\n")
fit_base <- as_cmdstan_fit(base_csvs)
cat("Reading M006...\n")
fit_m006 <- as_cmdstan_fit(m006_csvs)

dat_raw <- read_csv("/home/DCCS5/cerebellum_project/data/raw/behavioral_compilate.csv", show_col_types=FALSE)
test_subjs_10 <- head(unique(dat_raw$participant_id), 10)
test_dat <- dat_raw %>% filter(participant_id %in% test_subjs_10) %>%
    arrange(participant_id, ttp) %>% group_by(participant_id) %>%
    mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(F)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), subj_idx = as.integer(as.factor(participant_id)))

N_subj <- max(test_dat$subj_idx)
set.seed(42)
W_exp <- matrix(rnorm(N_subj * 32, 0, 1), nrow=N_subj, ncol=32)

cat("Extracting Summaries...\n")
sum_base <- fit_base$summary()
sum_m006 <- fit_m006$summary()

get_mean <- function(summ, pattern, n) {
    res <- numeric(n)
    for(i in 1:n) {
        row <- summ %>% filter(variable == paste0(pattern, "[", i, "]"))
        res[i] <- row$mean
    }
    return(res)
}

base_a <- get_mean(sum_base, "a", N_subj)
base_tnd <- get_mean(sum_base, "tnd", N_subj)
base_v <- get_mean(sum_base, "v_ctx", N_subj)
base_alpha <- get_mean(sum_base, "alpha_ctx", N_subj)

m006_a_raw <- get_mean(sum_m006, "a_base_raw", N_subj)
m006_tnd <- get_mean(sum_m006, "tnd", N_subj)
m006_v <- get_mean(sum_m006, "v_ctx", N_subj)
m006_alpha <- get_mean(sum_m006, "alpha_ctx", N_subj)
m006_alpha_pc <- get_mean(sum_m006, "alpha_pc", N_subj)
m006_gamma <- get_mean(sum_m006, "gamma_var", N_subj)
m006_golgi <- get_mean(sum_m006, "golgi_scale", N_subj)
m006_tau <- get_mean(sum_m006, "tau_decay", N_subj)
m006_wu <- get_mean(sum_m006, "w_u", N_subj)

cat("Simulating Trajectories...\n")
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

// [[Rcpp::export]]
DataFrame eval_metrics(int N_trials, int N_subj, IntegerVector subj, IntegerVector resp, NumericVector reward, NumericVector rt, NumericVector iti,
                       NumericVector b_a, NumericVector b_tnd, NumericVector b_v, NumericVector b_alpha,
                       NumericVector m_a_raw, NumericVector m_tnd, NumericVector m_v, NumericVector m_alpha, NumericVector m_apc, NumericVector m_g, NumericVector m_gs, NumericVector m_tau, NumericVector m_wu,
                       NumericMatrix W_exp) {
    
    NumericVector b_pred_rt(N_trials), b_pred_p1(N_trials);
    NumericVector m_pred_rt(N_trials), m_pred_p1(N_trials);
    
    std::vector<double> frac_alpha(32), kappa_vec(32);
    for(int i=0; i<32; ++i) {
        frac_alpha[i] = 0.1 + 0.8*(i/31.0);
        kappa_vec[i] = 0.1 + 0.89*(i/31.0);
    }
    
    std::vector<double> b_Q(N_subj * 2, 0.5);
    std::vector<double> m_Q(N_subj * 2, 0.5);
    std::vector<double> m_Q_diff(N_subj, 0.0);
    std::vector<std::vector<double>> frac_mem(N_subj, std::vector<double>(32, 0.0));
    std::vector<std::vector<double>> Z(N_subj, std::vector<double>(32, 0.0));
    std::vector<std::vector<double>> W_PC_latent(N_subj, std::vector<double>(32, 0.0));
    
    for(int t=0; t<N_trials; ++t) {
        int s = subj[t] - 1;
        int ch = resp[t] - 1;
        double R = reward[t];
        
        // BASELINE
        double b_veff = b_v[s] * (b_Q[s*2 + 1] - b_Q[s*2 + 0]);
        b_pred_p1[t] = 1.0 / (1.0 + std::exp(-2.0 * b_veff * b_a[s]));
        b_pred_rt[t] = b_tnd[s] + (b_a[s] / (2.0 * (b_veff != 0 ? std::abs(b_veff) : 1e-5))) * std::tanh(std::abs(b_veff) * b_a[s]);
        b_Q[s*2 + ch] += b_alpha[s] * (R - b_Q[s*2 + ch]);
        
        // M006
        double phys_decay = std::exp(-iti[t] / m_tau[s]);
        for(int i=0; i<32; ++i) {
            frac_mem[s][i] = frac_alpha[i] * frac_mem[s][i] + (1.0 - frac_alpha[i]) * W_exp(s, i) * m_Q[s*2 + ch];
            Z[s][i] = phys_decay * kappa_vec[i] * Z[s][i] + std::tanh(frac_mem[s][i]);
        }
        
        double cb0 = 0.0, cb1 = 0.0;
        std::vector<double> W_PC_eff(32), eff_z(32), S_mask(32);
        for(int i=0; i<32; ++i) {
            W_PC_eff[i] = 3.0 * std::tanh(W_PC_latent[s][i] / 3.0);
            eff_z[i] = W_PC_eff[i] * Z[s][i];
            S_mask[i] = std::tanh(m_gs[s] * std::sqrt(eff_z[i]*eff_z[i] + 1e-8));
            if(i < 16) cb0 += S_mask[i] * eff_z[i];
            else cb1 += S_mask[i] * eff_z[i];
        }
        
        double veff_scaled = (m_v[s] * 0.054) * m_Q_diff[s] + (m_g[s] * 0.054) * (cb1 - cb0);
        double m_veff = 18.51 * std::tanh(veff_scaled);
        
        double cb0_sq = cb0*cb0 + 1e-8, cb1_sq = cb1*cb1 + 1e-8;
        double a_raw = m_a_raw[s] + m_wu[s] * std::sqrt(cb0_sq * cb1_sq);
        double m_a_dyn = 0.11 + 7.36 * (1.0 / (1.0 + std::exp(-a_raw)));
        
        m_pred_p1[t] = 1.0 / (1.0 + std::exp(-2.0 * m_veff * m_a_dyn));
        m_pred_rt[t] = m_tnd[s] + (m_a_dyn / (2.0 * (m_veff != 0 ? std::abs(m_veff) : 1e-5))) * std::tanh(std::abs(m_veff) * m_a_dyn);
        
        double prev_E = R - m_Q[s*2 + ch];
        m_Q[s*2 + ch] += m_alpha[s] * prev_E;
        m_Q_diff[s] += (ch == 0 ? -1.0 : 1.0) * (m_alpha[s] * prev_E);
        
        double apc_E = m_apc[s] * prev_E;
        for(int i=0; i<32; ++i) {
            if(ch == 0 && i < 16) W_PC_latent[s][i] += apc_E * Z[s][i];
            if(ch == 1 && i >= 16) W_PC_latent[s][i] += apc_E * Z[s][i];
        }
    }
    
    return DataFrame::create(_["b_pred_p1"] = b_pred_p1, _["b_pred_rt"] = b_pred_rt, _["m_pred_p1"] = m_pred_p1, _["m_pred_rt"] = m_pred_rt);
}
')

preds <- eval_metrics(nrow(test_dat), N_subj, test_dat$subj_idx, test_dat$Boundary, test_dat$F, test_dat$RT, test_dat$ITI,
                      base_a, base_tnd, base_v, base_alpha,
                      m006_a_raw, m006_tnd, m006_v, m006_alpha, m006_alpha_pc, m006_gamma, m006_golgi, m006_tau, m006_wu,
                      W_exp)

test_dat$b_pred_p1 <- preds$b_pred_p1
test_dat$b_pred_rt <- preds$b_pred_rt
test_dat$m_pred_p1 <- preds$m_pred_p1
test_dat$m_pred_rt <- preds$m_pred_rt

calc_auc <- function(probs, truth) {
    if(sum(truth) == 0 || sum(!truth) == 0) return(NA)
    n1 <- sum(truth); n0 <- sum(!truth)
    u <- sum(rank(probs)[truth == 1]) - n1*(n1+1)/2
    return(u / (n1*n0))
}
calc_pr_auc <- function(probs, truth) {
    if(sum(truth) == 0 || sum(!truth) == 0) return(NA)
    ord <- order(probs, decreasing=TRUE)
    p <- probs[ord]; t <- truth[ord]
    tp <- cumsum(t); fp <- cumsum(!t)
    rec <- tp / sum(t); prec <- tp / (tp + fp)
    d_rec <- c(rec[1], diff(rec))
    return(sum(d_rec * prec, na.rm=TRUE))
}
calc_mcc <- function(probs, truth, thresh=0.5) {
    preds <- ifelse(probs >= thresh, 1, 0)
    tp <- sum(preds == 1 & truth == 1)
    tn <- sum(preds == 0 & truth == 0)
    fp <- sum(preds == 1 & truth == 0)
    fn <- sum(preds == 0 & truth == 1)
    denom <- sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if(denom == 0) return(0)
    return(((tp*tn) - (fp*fn)) / denom)
}

res_list <- list()
for(s in 1:N_subj) {
    s_dat <- test_dat %>% filter(subj_idx == s)
    s_dat <- s_dat %>% mutate(is_switch = ifelse(row_number()==1, FALSE, Boundary != lag(Boundary)),
                              b_p_switch = ifelse(lag(Boundary)==1, 1 - b_pred_p1, b_pred_p1),
                              m_p_switch = ifelse(lag(Boundary)==1, 1 - m_pred_p1, m_pred_p1)) %>% filter(row_number() > 1)
                              
    b_roc <- calc_auc(s_dat$b_p_switch, s_dat$is_switch)
    m_roc <- calc_auc(s_dat$m_p_switch, s_dat$is_switch)
    b_pr <- calc_pr_auc(s_dat$b_p_switch, s_dat$is_switch)
    m_pr <- calc_pr_auc(s_dat$m_p_switch, s_dat$is_switch)
    b_mcc <- calc_mcc(s_dat$b_p_switch, s_dat$is_switch)
    m_mcc <- calc_mcc(s_dat$m_p_switch, s_dat$is_switch)
    
    b_rmse <- sqrt(mean((s_dat$RT - s_dat$b_pred_rt)^2, na.rm=TRUE))
    m_rmse <- sqrt(mean((s_dat$RT - s_dat$m_pred_rt)^2, na.rm=TRUE))
    
    res_list[[length(res_list)+1]] <- data.frame(subj=s, model="Baseline", roc=b_roc, pr=b_pr, mcc=b_mcc, rmse=b_rmse)
    res_list[[length(res_list)+1]] <- data.frame(subj=s, model="M006", roc=m_roc, pr=m_pr, mcc=m_mcc, rmse=m_rmse)
}

df_res <- bind_rows(res_list) %>% filter(!is.na(roc))

cat("\nSummary Table:\n")
df_agg <- df_res %>% group_by(model) %>% summarise(ROC=mean(roc), PR=mean(pr), MCC=mean(mcc), RMSE=mean(rmse))
print(df_agg)

cat("\n--- Mixed Effect Models ---\n")
# We want to know if M006 > Baseline for ROC, PR, MCC, and if M006 < Baseline for RMSE
df_res$is_m006 <- ifelse(df_res$model == "M006", 1, 0)

# ROC
m_roc <- lmer(roc ~ is_m006 + (1|subj), data=df_res)
cat("\nROC-AUC Mixed Effect (is_m006 coefficient):\n")
print(summary(m_roc)$coefficients["is_m006",])
# Calculate effect size (Cohen's d)
# d = (Mean(M006) - Mean(Baseline)) / SD_pooled
sd_pooled_roc <- sd(df_res$roc)
d_roc <- summary(m_roc)$coefficients["is_m006", "Estimate"] / sd_pooled_roc
cat(sprintf("Effect Size (Cohen's d): %.3f\n", d_roc))

# PR
m_pr <- lmer(pr ~ is_m006 + (1|subj), data=df_res)
cat("\nPR-AUC Mixed Effect (is_m006 coefficient):\n")
print(summary(m_pr)$coefficients["is_m006",])
sd_pooled_pr <- sd(df_res$pr)
d_pr <- summary(m_pr)$coefficients["is_m006", "Estimate"] / sd_pooled_pr
cat(sprintf("Effect Size (Cohen's d): %.3f\n", d_pr))

# MCC
m_mcc <- lmer(mcc ~ is_m006 + (1|subj), data=df_res)
cat("\nMCC Mixed Effect (is_m006 coefficient):\n")
print(summary(m_mcc)$coefficients["is_m006",])
sd_pooled_mcc <- sd(df_res$mcc)
d_mcc <- summary(m_mcc)$coefficients["is_m006", "Estimate"] / sd_pooled_mcc
cat(sprintf("Effect Size (Cohen's d): %.3f\n", d_mcc))

# RMSE
m_rmse <- lmer(rmse ~ is_m006 + (1|subj), data=df_res)
cat("\nRT-RMSE Mixed Effect (is_m006 coefficient):\n")
print(summary(m_rmse)$coefficients["is_m006",])
sd_pooled_rmse <- sd(df_res$rmse)
d_rmse <- summary(m_rmse)$coefficients["is_m006", "Estimate"] / sd_pooled_rmse
cat(sprintf("Effect Size (Cohen's d): %.3f\n", d_rmse))
