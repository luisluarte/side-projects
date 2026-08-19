library(cmaes)
library(Rcpp)
library(dplyr)
library(survival)

cat("Loading dataset and sampling N=128...\n")
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(42)
participants <- unique(dat_all[['participant_id']])
num_participants <- length(participants)
dat_all <- dat_all[dat_all[['participant_id']] %in% participants, ]
dat_all$participant_factor <- as.integer(as.factor(dat_all$participant_id))

dat_all <- dat_all %>%
  group_by(participant_id) %>%
  arrange(ttp) %>%
  mutate(
    trial_idx = row_number(),
    n_trials = n(),
    is_test = ifelse(trial_idx > 0.7 * n_trials, 1, 0)
  ) %>%
  ungroup() %>%
  arrange(participant_factor, ttp)

cpp_code <- '
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

inline double wiener_pdf(double t_raw, int choice, double v, double a, double t_nd, double eps = 1e-9) {
  double t = t_raw - t_nd;
  if (t <= 0.001) return 1e-12;
  double sign = (choice == 1) ? 1.0 : -1.0;
  double w = 0.50; 
  double x0 = (choice == 1) ? (1.0 - w) : w;
  double drift_term = std::exp(sign * v * a * w - 0.5 * v * v * t);
  double tt = t / (a * a);
  double sum = 0.0;
  if (tt >= 0.08) {
    for (int k = 1; k <= 30; ++k) {
      double term = (double)k * std::sin((double)k * M_PI * x0) * std::exp(-0.5 * k * k * M_PI * M_PI * tt);
      sum += term;
      if (std::abs(term) < eps) break;
    }
    sum *= M_PI;
  } else {
    double sqrt_tt = std::sqrt(tt);
    for (int k = -15; k <= 15; ++k) {
      double num = (x0 + 2.0 * k);
      double term = num * std::exp(-0.5 * (num * num) / tt);
      sum += term;
    }
    sum /= (std::sqrt(2.0 * M_PI) * tt * sqrt_tt);
  }
  double pdf_val = (drift_term / (a * a)) * sum;
  return std::max(1e-12, pdf_val);
}

class SimpleRNG {
    uint32_t state;
public:
    SimpleRNG(uint32_t seed) : state(seed) {}
    uint32_t next() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
    double runif() { return (next() % 1000000) / 1000000.0; }
    double rnorm() {
        double u1 = std::max(1e-6, runif());
        double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
};

// M1 WSLS
// [[Rcpp::export]]
List evaluate_m1_cv(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& is_test_R, int num_participants) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2];
  double train_nll = 0.0; std::vector<double> out_prob1, out_pred_rt; std::vector<int> out_true_ch, out_true_out, out_subj;
  int prev_ch = -1; int prev_out = -1;
  for (int t=0; t<resp_R.size(); ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) { prev_ch = -1; prev_out = -1; }
    int ch = resp_R[t]; int out = out_R[t];
    int c_wsls = 1;
    if (prev_ch != -1) c_wsls = (prev_out == 1) ? prev_ch : ((prev_ch == 1) ? 2 : 1);
    double wsls_signal = (c_wsls == 1) ? 1.0 : -1.0;
    double v_t_ddm = beta_v * wsls_signal;
    double safe_v_t = std::abs(v_t_ddm) < 1e-4 ? (v_t_ddm >= 0 ? 1e-4 : -1e-4) : v_t_ddm;
    double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
    if (is_test_R[t] == 0) { train_nll -= std::log(dens); } 
    else {
      out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
      out_pred_rt.push_back(t_nd + (a / (2.0 * safe_v_t)) * std::tanh(a * safe_v_t / 2.0));
      out_true_ch.push_back(ch); out_true_out.push_back(out); out_subj.push_back(subj_idx_R[t]);
    }
    prev_ch = ch; prev_out = out;
  }
  return List::create(Named("train_nll") = train_nll, Named("prob1") = out_prob1, Named("pred_rt") = out_pred_rt, Named("true_ch") = out_true_ch, Named("true_out") = out_true_out, Named("subj") = out_subj);
}

// CFMR
// [[Rcpp::export]]
List evaluate_cfmr_cv(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& is_test_R, int num_participants) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4];
  double train_nll = 0.0; std::vector<double> out_prob1, out_pred_rt; std::vector<int> out_true_ch, out_true_out, out_subj;
  double Q1 = 0.0; double Q2 = 0.0;
  for (int t=0; t<resp_R.size(); ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) { Q1 = 0.0; Q2 = 0.0; }
    int ch = resp_R[t]; int out = out_R[t];
    double v_t_ddm = beta_v * (Q1 - Q2);
    double safe_v_t = std::abs(v_t_ddm) < 1e-4 ? (v_t_ddm >= 0 ? 1e-4 : -1e-4) : v_t_ddm;
    double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
    if (is_test_R[t] == 0) train_nll -= std::log(dens);
    else {
      out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
      out_pred_rt.push_back(t_nd + (a / (2.0 * safe_v_t)) * std::tanh(a * safe_v_t / 2.0));
      out_true_ch.push_back(ch); out_true_out.push_back(out); out_subj.push_back(subj_idx_R[t]);
    }
    if (ch == 1) { Q1 += (out == 1) ? eta_LTP * (1.0 - Q1) : eta_LTD * (-1.0 - Q1); } 
    else         { Q2 += (out == 1) ? eta_LTP * (1.0 - Q2) : eta_LTD * (-1.0 - Q2); }
  }
  return List::create(Named("train_nll") = train_nll, Named("prob1") = out_prob1, Named("pred_rt") = out_pred_rt, Named("true_ch") = out_true_ch, Named("true_out") = out_true_out, Named("subj") = out_subj);
}

// NGRC CFMR
// [[Rcpp::export]]
List evaluate_ngrc_cfmr_cv(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& is_test_R, int num_participants) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4];
  double train_nll = 0.0; std::vector<double> out_prob1, out_pred_rt; std::vector<int> out_true_ch, out_true_out, out_subj;
  int N_GC = 50; std::vector<double> W_PF1(N_GC, 0.0); std::vector<double> W_PF2(N_GC, 0.0);
  std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(3, 0.0));
  int hist_c1 = 0; int hist_c2 = 0; int hist_c3 = 0;
  for (int t=0; t<resp_R.size(); ++t) {
    int s_idx = subj_idx_R[t];
    if (t == 0 || subj_idx_R[t] != subj_idx_R[t-1]) {
        std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
        hist_c1 = 0; hist_c2 = 0; hist_c3 = 0;
        SimpleRNG rng(s_idx + 42);
        for (int i=0; i<N_GC; ++i) { for (int j=0; j<3; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt(3.0); }
    }
    int ch = resp_R[t]; int out = out_R[t];
    std::vector<double> h_t(N_GC, 0.0); double Q1 = 0.0; double Q2 = 0.0;
    for (int i=0; i<N_GC; ++i) {
        h_t[i] = std::tanh(W_MF_GC[i][0] * hist_c1 + W_MF_GC[i][1] * hist_c2 + W_MF_GC[i][2] * hist_c3);
        Q1 += W_PF1[i] * h_t[i]; Q2 += W_PF2[i] * h_t[i];
    }
    double v_t_ddm = beta_v * (Q1 - Q2);
    double safe_v_t = std::abs(v_t_ddm) < 1e-4 ? (v_t_ddm >= 0 ? 1e-4 : -1e-4) : v_t_ddm;
    double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
    if (is_test_R[t] == 0) train_nll -= std::log(dens);
    else {
      out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
      out_pred_rt.push_back(t_nd + (a / (2.0 * safe_v_t)) * std::tanh(a * safe_v_t / 2.0));
      out_true_ch.push_back(ch); out_true_out.push_back(out); out_subj.push_back(s_idx);
    }
    if (ch == 1) {
        for (int i=0; i<N_GC; ++i) W_PF1[i] += (out == 1) ? eta_LTP * (1.0 - Q1) * h_t[i] : eta_LTD * (-1.0 - Q1) * h_t[i];
    } else {
        for (int i=0; i<N_GC; ++i) W_PF2[i] += (out == 1) ? eta_LTP * (1.0 - Q2) * h_t[i] : eta_LTD * (-1.0 - Q2) * h_t[i];
    }
    hist_c3 = hist_c2; hist_c2 = hist_c1; hist_c1 = (ch == 1) ? 1 : -1;
  }
  return List::create(Named("train_nll") = train_nll, Named("prob1") = out_prob1, Named("pred_rt") = out_pred_rt, Named("true_ch") = out_true_ch, Named("true_out") = out_true_out, Named("subj") = out_subj);
}
'
sourceCpp(code = cpp_code)

calc_te <- function(df) {
    # TE(Out -> PredCh | PrevPredCh)
    df <- df %>% filter(!is.na(prev_out) & !is.na(prev_pred_ch))
    
    # Joint P(X=prev_out, Y=pred_ch, Z=prev_pred_ch)
    joint_counts <- df %>% group_by(prev_out, pred_ch, prev_pred_ch) %>% summarize(n = n(), .groups="drop")
    joint_counts$P_xyz <- joint_counts$n / sum(joint_counts$n)
    
    # Marginal P(X, Z)
    xz_counts <- df %>% group_by(prev_out, prev_pred_ch) %>% summarize(n_xz = n(), .groups="drop")
    xz_counts$P_xz <- xz_counts$n_xz / sum(xz_counts$n_xz)
    
    # Marginal P(Z)
    z_counts <- df %>% group_by(prev_pred_ch) %>% summarize(n_z = n(), .groups="drop")
    z_counts$P_z <- z_counts$n_z / sum(z_counts$n_z)
    
    # Marginal P(Y, Z)
    yz_counts <- df %>% group_by(pred_ch, prev_pred_ch) %>% summarize(n_yz = n(), .groups="drop")
    yz_counts$P_yz <- yz_counts$n_yz / sum(yz_counts$n_yz)
    
    merged <- joint_counts %>%
        left_join(xz_counts, by=c("prev_out", "prev_pred_ch")) %>%
        left_join(z_counts, by="prev_pred_ch") %>%
        left_join(yz_counts, by=c("pred_ch", "prev_pred_ch"))
        
    merged <- merged %>% mutate(
        P_y_given_xz = P_xyz / P_xz,
        P_y_given_z = P_yz / P_z,
        TE_term = P_xyz * log2(P_y_given_xz / P_y_given_z)
    )
    
    return(sum(merged$TE_term, na.rm=TRUE))
}

compute_advanced_metrics <- function(res, true_rt_global) {
    p1 <- res$prob1
    set.seed(42)
    preds <- ifelse(runif(length(p1)) < p1, 1, 2)
    true_ch <- res$true_ch
    pred_rt <- res$pred_rt
    
    # 1. Wasserstein 1D (Mean Absolute Difference of Sorted Quantiles)
    emp_rt_sorted <- sort(true_rt_global)
    pred_rt_sorted <- sort(pred_rt)
    wasserstein_dist <- mean(abs(emp_rt_sorted - pred_rt_sorted))
    
    # 2. Transfer Entropy (Out -> PredCh)
    df <- data.frame(subj = res$subj, true_ch = true_ch, pred_ch = preds, out = res$true_out)
    df <- df %>% group_by(subj) %>% mutate(
        prev_out = lag(out),
        prev_pred_ch = lag(pred_ch)
    ) %>% ungroup()
    
    te_val <- calc_te(df)
    
    # 3. Ljung-Box Test on Residuals
    df$res <- ifelse(df$true_ch == 1, 1, 0) - p1
    lb_stats <- df %>% group_by(subj) %>% summarize(
        lb_stat = Box.test(res, lag=5, type="Ljung-Box")$statistic
    ) %>% pull(lb_stat)
    mean_lb_stat <- mean(lb_stats, na.rm=TRUE)
    
    # 4. Hazard Function MIAE
    sf_emp <- survfit(Surv(true_rt_global, rep(1, length(true_rt_global))) ~ 1)
    sf_pred <- survfit(Surv(pred_rt, rep(1, length(pred_rt))) ~ 1)
    
    t_grid <- seq(0.1, 3.0, by=0.05)
    haz_emp <- stepfun(sf_emp$time, c(0, -log(sf_emp$surv)))(t_grid)
    haz_pred <- stepfun(sf_pred$time, c(0, -log(sf_pred$surv)))(t_grid)
    
    valid_idx <- is.finite(haz_emp) & is.finite(haz_pred)
    haz_miae <- mean(abs(haz_emp[valid_idx] - haz_pred[valid_idx]), na.rm=TRUE)
    
    return(c(Wasserstein = wasserstein_dist, Transfer_Entropy = te_val, Ljung_Box_Q = mean_lb_stat, Hazard_MIAE = haz_miae))
}

cat("\nOptimizing M1 (3 parameters)...\n")
obj_m1 <- function(phi) {
    if (any(phi < c(0.1, 0.01, 0.0)) || any(phi > c(5.0, 1.0, 10.0))) return(1e9)
    res <- evaluate_m1_cv(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)
    return(res$train_nll)
}
opt_m1 <- cma_es(c(1.0, 0.3, 2.0), obj_m1, lower=c(0.1, 0.01, 0.0), upper=c(5.0, 1.0, 10.0), control=list(maxit=50, trace=FALSE))
res_m1 <- evaluate_m1_cv(opt_m1$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)

# Extract true_rt aligned with test data
true_rt_global <- dat_all$RT[dat_all$is_test == 1]
met_m1 <- compute_advanced_metrics(res_m1, true_rt_global)

cat("Optimizing CFMR (5 parameters)...\n")
obj_cfmr <- function(phi) {
    if (any(phi < c(0.1, 0.01, 0.0, 0.0, 0.0)) || any(phi > c(5.0, 1.0, 10.0, 1.0, 1.0))) return(1e9)
    res <- evaluate_cfmr_cv(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)
    return(res$train_nll)
}
opt_cfmr <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5), obj_cfmr, lower=c(0.1, 0.01, 0.0, 0.0, 0.0), upper=c(5.0, 1.0, 10.0, 1.0, 1.0), control=list(maxit=50, trace=FALSE))
res_cfmr <- evaluate_cfmr_cv(opt_cfmr$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)
met_cfmr <- compute_advanced_metrics(res_cfmr, true_rt_global)

cat("Optimizing NGRC CFMR (5 parameters)...\n")
obj_ngrc <- function(phi) {
    if (any(phi < c(0.1, 0.01, 0.0, 0.0, 0.0)) || any(phi > c(5.0, 1.0, 10.0, 1.0, 1.0))) return(1e9)
    res <- evaluate_ngrc_cfmr_cv(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)
    return(res$train_nll)
}
opt_ngrc <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5), obj_ngrc, lower=c(0.1, 0.01, 0.0, 0.0, 0.0), upper=c(5.0, 1.0, 10.0, 1.0, 1.0), control=list(maxit=50, trace=FALSE))
res_ngrc <- evaluate_ngrc_cfmr_cv(opt_ngrc$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)
met_ngrc <- compute_advanced_metrics(res_ngrc, true_rt_global)

cat("\n=======================================================\n")
cat("          ADVANCED METRICS COMPARISON                  \n")
cat("=======================================================\n")

df_out <- data.frame(
    Model = c("M1 (WSLS)", "CFMR (Scalar)", "NGRC CFMR"),
    Wasserstein_W1 = c(met_m1["Wasserstein"], met_cfmr["Wasserstein"], met_ngrc["Wasserstein"]),
    Transfer_Entropy = c(met_m1["Transfer_Entropy"], met_cfmr["Transfer_Entropy"], met_ngrc["Transfer_Entropy"]),
    Ljung_Box_Q = c(met_m1["Ljung_Box_Q"], met_cfmr["Ljung_Box_Q"], met_ngrc["Ljung_Box_Q"]),
    Hazard_MIAE = c(met_m1["Hazard_MIAE"], met_cfmr["Hazard_MIAE"], met_ngrc["Hazard_MIAE"])
)

print(df_out, row.names = FALSE)
cat("=======================================================\n")
write.csv(df_out, "results/tables/advanced_metrics_comparison.csv", row.names=FALSE)
