library(cmaes)
library(Rcpp)
library(dplyr)
library(PRROC)

cat("Loading dataset...\n")
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 50)
num_participants <- length(participants)
dat_all <- dat_all[dat_all[['participant_id']] %in% participants, ]
dat_all$participant_factor <- as.integer(as.factor(dat_all$participant_id))

dat_all <- dat_all %>%
  group_by(participant_id) %>%
  arrange(ttp) %>%
  mutate(trial_idx = row_number() - 1) %>%
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

// Pure CFMR Baseline
// [[Rcpp::export]]
List eval_cfmr(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4];
  double nll = 0.0; std::vector<double> out_prob1; 
  double Q1 = 0.0; double Q2 = 0.0;
  for (int t=0; t<resp_R.size(); ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) { Q1 = 0.0; Q2 = 0.0; }
    int ch = resp_R[t]; int out = out_R[t];
    double v_t = beta_v * (Q1 - Q2);
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    nll -= std::log(wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd));
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    if (ch == 1) { Q1 += (out == 1) ? eta_LTP * (1.0 - Q1) : eta_LTD * (-1.0 - Q1); } 
    else         { Q2 += (out == 1) ? eta_LTP * (1.0 - Q2) : eta_LTD * (-1.0 - Q2); }
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1);
}

// Architecture A: Residual Error Correction
// Cortex: CFMR. Cerebellum: 20:400:80 predicting Cortical Residual (R - Q_CTX).
// [[Rcpp::export]]
List eval_arch_a(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4]; double w_cb = phi[5];
  double nll = 0.0; std::vector<double> out_prob1; 
  
  double Q1_CTX = 0.0; double Q2_CTX = 0.0;
  int N_MF = 20; int N_GC = 400; int N_MLI = 80;
  std::vector<double> W_PF1(N_GC, 0.0); std::vector<double> W_PF2(N_GC, 0.0);
  std::vector<double> W_MLI1(N_MLI, 0.0); std::vector<double> W_MLI2(N_MLI, 0.0);
  std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
  std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
  std::vector<double> mf(N_MF, 0.0);
  
  for (int t=0; t<resp_R.size(); ++t) {
    int s_idx = subj_idx_R[t];
    if (t == 0 || subj_idx_R[t] != subj_idx_R[t-1]) {
        Q1_CTX = 0.0; Q2_CTX = 0.0;
        std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
        std::fill(W_MLI1.begin(), W_MLI1.end(), 0.0); std::fill(W_MLI2.begin(), W_MLI2.end(), 0.0);
        std::fill(mf.begin(), mf.end(), 0.0);
        SimpleRNG rng(s_idx + 42);
        for (int i=0; i<N_GC; ++i) { for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF); }
        for (int i=0; i<N_MLI; ++i) { for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC); }
    }
    
    int ch = resp_R[t]; int out = out_R[t]; double R_raw = (out == 1) ? 1.0 : -1.0;
    
    std::vector<double> gc(N_GC, 0.0);
    for (int i=0; i<N_GC; ++i) {
        double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
        gc[i] = std::tanh(act);
    }
    std::vector<double> mli(N_MLI, 0.0);
    for (int i=0; i<N_MLI; ++i) {
        double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
        mli[i] = std::tanh(act);
    }
    
    double Q1_CB = 0.0; double Q2_CB = 0.0;
    for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
    for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
    
    double v_t = beta_v * ( (Q1_CTX + w_cb*Q1_CB) - (Q2_CTX + w_cb*Q2_CB) );
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    nll -= std::log(wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd));
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    
    double lr = (out == 1) ? eta_LTP : eta_LTD;
    double lr_cb = lr / (double)N_GC;
    double lr_mli = lr / (double)N_MLI;
    if (ch == 1) {
        double delta_ctx = R_raw - Q1_CTX;
        Q1_CTX += (out == 1) ? eta_LTP * (1.0 - Q1_CTX) : eta_LTD * (-1.0 - Q1_CTX);
        double delta_cb = delta_ctx - Q1_CB;
        for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * delta_cb * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * delta_cb * mli[i];
    } else {
        double delta_ctx = R_raw - Q2_CTX;
        Q2_CTX += (out == 1) ? eta_LTP * (1.0 - Q2_CTX) : eta_LTD * (-1.0 - Q2_CTX);
        double delta_cb = delta_ctx - Q2_CB;
        for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * delta_cb * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * delta_cb * mli[i];
    }
    
    for(int k=8; k>=0; --k) { mf[k+1] = mf[k]; mf[10+k+1] = mf[10+k]; }
    mf[0] = (ch == 1) ? 1.0 : -1.0; mf[10] = R_raw;
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1);
}

// Architecture B: Systems Consolidation (Cerebellum Teaches Cortex)
// [[Rcpp::export]]
List eval_arch_b(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4]; double w_teach = phi[5];
  double nll = 0.0; std::vector<double> out_prob1; 
  
  double Q1_CTX = 0.0; double Q2_CTX = 0.0;
  int N_MF = 20; int N_GC = 400; int N_MLI = 80;
  std::vector<double> W_PF1(N_GC, 0.0); std::vector<double> W_PF2(N_GC, 0.0);
  std::vector<double> W_MLI1(N_MLI, 0.0); std::vector<double> W_MLI2(N_MLI, 0.0);
  std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
  std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
  std::vector<double> mf(N_MF, 0.0);
  
  for (int t=0; t<resp_R.size(); ++t) {
    int s_idx = subj_idx_R[t];
    if (t == 0 || subj_idx_R[t] != subj_idx_R[t-1]) {
        Q1_CTX = 0.0; Q2_CTX = 0.0;
        std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
        std::fill(W_MLI1.begin(), W_MLI1.end(), 0.0); std::fill(W_MLI2.begin(), W_MLI2.end(), 0.0);
        std::fill(mf.begin(), mf.end(), 0.0);
        SimpleRNG rng(s_idx + 42);
        for (int i=0; i<N_GC; ++i) { for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF); }
        for (int i=0; i<N_MLI; ++i) { for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC); }
    }
    
    int ch = resp_R[t]; int out = out_R[t]; double R_raw = (out == 1) ? 1.0 : -1.0;
    
    std::vector<double> gc(N_GC, 0.0);
    for (int i=0; i<N_GC; ++i) {
        double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
        gc[i] = std::tanh(act);
    }
    std::vector<double> mli(N_MLI, 0.0);
    for (int i=0; i<N_MLI; ++i) {
        double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
        mli[i] = std::tanh(act);
    }
    
    double Q1_CB = 0.0; double Q2_CB = 0.0;
    for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
    for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
    
    double v_t = beta_v * (Q1_CTX - Q2_CTX);
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    nll -= std::log(wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd));
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    
    double lr = (out == 1) ? eta_LTP : eta_LTD;
    double lr_cb = lr / (double)N_GC;
    double lr_mli = lr / (double)N_MLI;
    if (ch == 1) {
        double target = (1.0 - w_teach) * R_raw + w_teach * Q1_CB;
        Q1_CTX += lr * (target - Q1_CTX);
        double err_cb = R_raw - Q1_CB;
        for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
    } else {
        double target = (1.0 - w_teach) * R_raw + w_teach * Q2_CB;
        Q2_CTX += lr * (target - Q2_CTX);
        double err_cb = R_raw - Q2_CB;
        for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
    }
    
    for(int k=8; k>=0; --k) { mf[k+1] = mf[k]; mf[10+k+1] = mf[10+k]; }
    mf[0] = (ch == 1) ? 1.0 : -1.0; mf[10] = R_raw;
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1);
}

// Architecture C: Additive Parallel
// [[Rcpp::export]]
List eval_arch_c(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4]; double w_cb = phi[5];
  double nll = 0.0; std::vector<double> out_prob1; 
  
  double Q1_CTX = 0.0; double Q2_CTX = 0.0;
  int N_MF = 20; int N_GC = 400; int N_MLI = 80;
  std::vector<double> W_PF1(N_GC, 0.0); std::vector<double> W_PF2(N_GC, 0.0);
  std::vector<double> W_MLI1(N_MLI, 0.0); std::vector<double> W_MLI2(N_MLI, 0.0);
  std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
  std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
  std::vector<double> mf(N_MF, 0.0);
  
  for (int t=0; t<resp_R.size(); ++t) {
    int s_idx = subj_idx_R[t];
    if (t == 0 || subj_idx_R[t] != subj_idx_R[t-1]) {
        Q1_CTX = 0.0; Q2_CTX = 0.0;
        std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
        std::fill(W_MLI1.begin(), W_MLI1.end(), 0.0); std::fill(W_MLI2.begin(), W_MLI2.end(), 0.0);
        std::fill(mf.begin(), mf.end(), 0.0);
        SimpleRNG rng(s_idx + 42);
        for (int i=0; i<N_GC; ++i) { for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF); }
        for (int i=0; i<N_MLI; ++i) { for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC); }
    }
    
    int ch = resp_R[t]; int out = out_R[t]; double R_raw = (out == 1) ? 1.0 : -1.0;
    
    std::vector<double> gc(N_GC, 0.0);
    for (int i=0; i<N_GC; ++i) {
        double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
        gc[i] = std::tanh(act);
    }
    std::vector<double> mli(N_MLI, 0.0);
    for (int i=0; i<N_MLI; ++i) {
        double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
        mli[i] = std::tanh(act);
    }
    
    double Q1_CB = 0.0; double Q2_CB = 0.0;
    for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
    for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
    
    double v_t = beta_v * ( (1.0 - w_cb)*Q1_CTX + w_cb*Q1_CB - ((1.0 - w_cb)*Q2_CTX + w_cb*Q2_CB) );
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    nll -= std::log(wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd));
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    
    double lr = (out == 1) ? eta_LTP : eta_LTD;
    double lr_cb = lr / (double)N_GC;
    double lr_mli = lr / (double)N_MLI;
    if (ch == 1) {
        Q1_CTX += (out == 1) ? eta_LTP * (1.0 - Q1_CTX) : eta_LTD * (-1.0 - Q1_CTX);
        double err_cb = R_raw - Q1_CB;
        for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
    } else {
        Q2_CTX += (out == 1) ? eta_LTP * (1.0 - Q2_CTX) : eta_LTD * (-1.0 - Q2_CTX);
        double err_cb = R_raw - Q2_CB;
        for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
    }
    
    for(int k=8; k>=0; --k) { mf[k+1] = mf[k]; mf[10+k+1] = mf[10+k]; }
    mf[0] = (ch == 1) ? 1.0 : -1.0; mf[10] = R_raw;
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1);
}
'
sourceCpp(code = cpp_code)

cat("Optimizing Pure CFMR Baseline...\n")
opt_cfmr <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5), 
                   function(p) { if(any(p<c(0.1,0.01,0,0,0))||any(p>c(5,1,10,1,1))) 1e9 else eval_cfmr(p, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT)$nll }, 
                   lower=c(0.1, 0.01, 0, 0, 0), upper=c(5, 1, 10, 1, 1), control=list(maxit=50))

cat("Optimizing Architecture A: Residual Error Correction...\n")
opt_a <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5, 0.5), 
                function(p) { if(any(p<c(0.1,0.01,0,0,0,0))||any(p>c(5,1,10,1,1,2))) 1e9 else eval_arch_a(p, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT)$nll }, 
                lower=c(0.1, 0.01, 0, 0, 0, 0), upper=c(5, 1, 10, 1, 1, 2), control=list(maxit=50))

cat("Optimizing Architecture B: Systems Consolidation...\n")
opt_b <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5, 0.5), 
                function(p) { if(any(p<c(0.1,0.01,0,0,0,0))||any(p>c(5,1,10,1,1,1))) 1e9 else eval_arch_b(p, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT)$nll }, 
                lower=c(0.1, 0.01, 0, 0, 0, 0), upper=c(5, 1, 10, 1, 1, 1), control=list(maxit=50))

cat("Optimizing Architecture C: Additive Parallel...\n")
opt_c <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5, 0.5), 
                function(p) { if(any(p<c(0.1,0.01,0,0,0,0))||any(p>c(5,1,10,1,1,1))) 1e9 else eval_arch_c(p, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT)$nll }, 
                lower=c(0.1, 0.01, 0, 0, 0, 0), upper=c(5, 1, 10, 1, 1, 1), control=list(maxit=50))

res_cfmr <- eval_cfmr(opt_cfmr$par, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT)
res_a <- eval_arch_a(opt_a$par, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT)
res_b <- eval_arch_b(opt_b$par, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT)
res_c <- eval_arch_c(opt_c$par, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT)

dat_all$p1_cfmr <- res_cfmr$prob1
dat_all$p1_a <- res_a$prob1
dat_all$p1_b <- res_b$prob1
dat_all$p1_c <- res_c$prob1

# Subject-level metrics
subj_res <- dat_all %>% group_by(participant_id) %>% summarize(
    nll_cfmr = -sum(log(ifelse(Resp==1, p1_cfmr, 1-p1_cfmr))),
    nll_a = -sum(log(ifelse(Resp==1, p1_a, 1-p1_a))),
    nll_b = -sum(log(ifelse(Resp==1, p1_b, 1-p1_b))),
    nll_c = -sum(log(ifelse(Resp==1, p1_c, 1-p1_c))),
    prauc_cfmr = pr.curve(scores.class0 = p1_cfmr, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral,
    prauc_a = pr.curve(scores.class0 = p1_a, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral,
    prauc_b = pr.curve(scores.class0 = p1_b, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral,
    prauc_c = pr.curve(scores.class0 = p1_c, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral
)

cat("\n=======================================================\n")
cat("          UNIFIED MODEL SEARCH RESULTS                 \n")
cat("=======================================================\n")

safe_t_test <- function(x, y, alt) {
    res <- tryCatch(t.test(x, y, paired=TRUE, alternative=alt), error = function(e) NULL)
    if (is.null(res)) return(list(p.value = 1.0))
    return(res)
}

tt_nll_a <- safe_t_test(subj_res$nll_cfmr, subj_res$nll_a, "greater") 
tt_nll_b <- safe_t_test(subj_res$nll_cfmr, subj_res$nll_b, "greater")
tt_nll_c <- safe_t_test(subj_res$nll_cfmr, subj_res$nll_c, "greater")

tt_pr_a <- safe_t_test(subj_res$prauc_a, subj_res$prauc_cfmr, "greater") 
tt_pr_b <- safe_t_test(subj_res$prauc_b, subj_res$prauc_cfmr, "greater")
tt_pr_c <- safe_t_test(subj_res$prauc_c, subj_res$prauc_cfmr, "greater")

res_df <- data.frame(
    Model = c("Arch A: Residual", "Arch B: Consolidation", "Arch C: Additive"),
    Total_NLL = c(sum(subj_res$nll_a), sum(subj_res$nll_b), sum(subj_res$nll_c)),
    Mean_PRAUC = c(mean(subj_res$prauc_a, na.rm=TRUE), mean(subj_res$prauc_b, na.rm=TRUE), mean(subj_res$prauc_c, na.rm=TRUE)),
    NLL_PValue = c(tt_nll_a$p.value, tt_nll_b$p.value, tt_nll_c$p.value),
    PRAUC_PValue = c(tt_pr_a$p.value, tt_pr_b$p.value, tt_pr_c$p.value)
)
print(res_df, row.names=FALSE)
cat(sprintf("Baseline CFMR Total NLL: %.2f | Mean PR-AUC: %.4f\n", sum(subj_res$nll_cfmr), mean(subj_res$prauc_cfmr)))
cat("=======================================================\n")
